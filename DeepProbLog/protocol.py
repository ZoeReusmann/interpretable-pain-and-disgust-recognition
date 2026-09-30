import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

"""
Ziel: Protokoll über die DeepProbLog-Trainingsläufe führen (Modell A / Modell B / Modell C).

Analog zum FastLAS-Runner aus dem ILP-Teil: wrappt train.py als Subprozess,
protokolliert Start/Ende/Laufzeit/Status robust auch bei Abstürzen oder
manuellem Abbruch (Ctrl+C), und hängt die Ergebnisse an
DeepProbLog/results/experiment_log.csv an. train.py selbst loggt nichts -
alles Protokollieren passiert hier, train.py bleibt ein einfacher, isoliert
testbarer Trainings-Baustein.

Was kommt rein? steht in LOG_COLUMNS.
Ergebnisse in DeepProbLog/results/experiment_log.csv

Neben Accuracy werden zusätzlich Precision, Recall, F1 und Specificity
protokolliert (gleiche Kennzahlen wie im CNN-Teil), damit die Ergebnisse
über die gesamte Pipeline hinweg vergleichbar sind.
"""

RESULTS_DIR = Path("DeepProbLog/results")
LOG_FILE = RESULTS_DIR / "experiment_log.csv"
TRAIN_SCRIPT_DEFAULT = "DeepProbLog/train.py"

LOG_COLUMNS = [
    "run_id",
    "name",
    "model",
    "dataset",
    "epochs",
    "learning_rate",
    "start_time",
    "end_time",
    "runtime_seconds",
    "exit_code",
    "status",
    "learned_parameters",
    "final_loss",
    "train_accuracy",
    "test_accuracy",
    "train_precision",
    "train_recall",
    "train_f1",
    "train_specificity",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_specificity",
    "output_file",
    "heartbeat_file",
    "crash_file",
    "notes",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepProbLog-Trainingslauf ausfuehren und automatisch protokollieren."
    )

    parser.add_argument("--model", required=True, choices=["model_a", "model_b", "model_c"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--notes", default="")
    parser.add_argument("--train-script", default=TRAIN_SCRIPT_DEFAULT)

    return parser.parse_args()


def append_heartbeat(
    heartbeat_file: Path,
    stop_event: threading.Event,
) -> None:
    """Schreibt alle 60 Sekunden einen Zeitstempel + Speicherauslastung in die
    Heartbeat-Datei, solange der Trainingslauf noch aktiv ist."""
    while not stop_event.wait(60):
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

        try:
            memory = subprocess.run(
                ["free", "-h"],
                capture_output=True,
                text=True,
                check=False,
            )
            memory_line = next(
                (line.strip() for line in memory.stdout.splitlines() if line.startswith("Mem:")),
                "Mem: unbekannt",
            )
        except Exception as exc:
            memory_line = f"Mem: Fehler beim Abfragen ({exc})"

        with heartbeat_file.open("a", encoding="utf-8") as file:
            file.write(f"{timestamp} | {memory_line}\n")


def determine_status(exit_code: int, output_file: Path) -> str:
    if exit_code != 0:
        return "FAILED"

    if not output_file.exists() or output_file.stat().st_size == 0:
        return "EMPTY_OUTPUT"

    output = output_file.read_text(encoding="utf-8", errors="replace")

    if "Modell gespeichert" not in output:
        return "INCOMPLETE"

    return "SUCCESS"


def extract_metrics(output_file: Path, status: str) -> dict:
    """Liest die von train.py ausgegebenen Kennzahlen aus dem mitgeschriebenen
    Output (analog zu count_rules() im ILP-Runner, der die FastLAS-Ausgabe parst).

    WICHTIG: Die Segmente werden auf den jeweiligen Abschnitt BEGRENZT (TRAIN
    endet dort, wo TEST beginnt) - sonst würde z.B. eine Test-Accuracy versehentlich
    als Train-Accuracy gematcht werden, falls die Reihenfolge im Output variiert.
    """
    empty_result = {col: "" for col in LOG_COLUMNS if col not in (
        "run_id", "name", "model", "dataset", "epochs", "learning_rate",
        "start_time", "end_time", "runtime_seconds", "exit_code", "status",
        "output_file", "heartbeat_file", "crash_file", "notes",
    )}

    if not output_file.exists():
        return empty_result

    text = output_file.read_text(encoding="utf-8", errors="replace")

    def bounded_segment(start_marker: str, end_marker: str = None) -> str:
        start_idx = text.find(start_marker)
        if start_idx == -1:
            return ""
        if end_marker:
            end_idx = text.find(end_marker, start_idx)
            if end_idx == -1:
                return text[start_idx:]
            return text[start_idx:end_idx]
        return text[start_idx:]

    def last_match_in(segment: str, pattern: str):
        matches = re.findall(pattern, segment)
        return matches[-1] if matches else ""

    learned_parameters = ""
    params_match = re.search(r"Gelernte Parameter.*?:\s*\n?(\[.*?\])", text, re.DOTALL)
    if params_match:
        learned_parameters = params_match.group(1)

    final_loss = last_match_in(text, r"Loss[:\s]+([\d.eE+-]+)")

    train_segment = bounded_segment("--- Accuracy auf TRAIN ---", "--- Accuracy auf TEST")
    test_segment = bounded_segment("--- Accuracy auf TEST")

    train_accuracy = last_match_in(train_segment, r"Accuracy[:\s]+([\d.]+)")
    test_accuracy = last_match_in(test_segment, r"Accuracy[:\s]+([\d.]+)")

    train_precision = last_match_in(train_segment, r"Precision:\s*([\d.]+)")
    train_recall = last_match_in(train_segment, r"Recall:\s*([\d.]+)")
    train_f1 = last_match_in(train_segment, r"F1:\s*([\d.]+)")
    train_specificity = last_match_in(train_segment, r"Specificity:\s*([\d.]+)")

    test_precision = last_match_in(test_segment, r"Precision:\s*([\d.]+)")
    test_recall = last_match_in(test_segment, r"Recall:\s*([\d.]+)")
    test_f1 = last_match_in(test_segment, r"F1:\s*([\d.]+)")
    test_specificity = last_match_in(test_segment, r"Specificity:\s*([\d.]+)")

    return {
        "learned_parameters": learned_parameters,
        "final_loss": final_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "train_specificity": train_specificity,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_specificity": test_specificity,
    }


def append_to_log(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()

    with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_arguments()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.name}"
    output_file = RESULTS_DIR / f"{run_id}_output.txt"
    heartbeat_file = RESULTS_DIR / f"{run_id}_heartbeat.log"
    crash_file = RESULTS_DIR / f"{run_id}_crash.log"

    print(f"Starte Experiment: {args.name}")
    print(f"Modell: {args.model}")
    print(f"Output: {output_file}")

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=append_heartbeat, args=(heartbeat_file, stop_event), daemon=True
    )
    heartbeat_thread.start()

    start_time = datetime.now().astimezone()
    start_perf = time.perf_counter()

    cmd = [
        sys.executable,
        args.train_script,
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
    ]

    exit_code = 0
    process = None

    def handle_interrupt(signum, frame):
        if process is not None:
            process.terminate()
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGINT, handle_interrupt)

    try:
        with output_file.open("w", encoding="utf-8") as out:
            process = subprocess.run(
                cmd,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        exit_code = process.returncode
    except KeyboardInterrupt:
        exit_code = 130
        crash_file.write_text("Manuell abgebrochen (Ctrl+C).\n", encoding="utf-8")
    except Exception as exc:
        exit_code = 1
        crash_file.write_text(f"Unerwarteter Fehler: {exc}\n", encoding="utf-8")
    finally:
        signal.signal(signal.SIGINT, old_handler)
        stop_event.set()
        heartbeat_thread.join(timeout=5)

    end_time = datetime.now().astimezone()
    runtime_seconds = round(time.perf_counter() - start_perf, 2)

    status = determine_status(exit_code, output_file)
    metrics = extract_metrics(output_file, status)

    row = {
        "run_id": run_id,
        "name": args.name,
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "start_time": start_time.isoformat(timespec="seconds"),
        "end_time": end_time.isoformat(timespec="seconds"),
        "runtime_seconds": runtime_seconds,
        "exit_code": exit_code,
        "status": status,
        "output_file": str(output_file),
        "heartbeat_file": str(heartbeat_file),
        "crash_file": str(crash_file) if crash_file.exists() else "",
        "notes": args.notes,
        **metrics,
    }

    append_to_log(row)

    print(f"\nStatus: {status}")
    print(f"Laufzeit: {runtime_seconds} Sekunden")
    print(f"Train-Accuracy: {metrics.get('train_accuracy', '')}")
    print(f"Test-Accuracy: {metrics.get('test_accuracy', '')}")
    print(f"Protokoll: {LOG_FILE}")


if __name__ == "__main__":
    main()