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
Ziel: Protokoll über die Runs führen

Was kommt rein? steht in LOG_COLUMNS
Ergebnisse in results/experiment_log.csv

Run starten in der Form

python scripts/run_experiment.py \
  --task oracle/task_pain_oracle_tiny_penalty-uniform1.las \
  --name pain_oracle_tiny_uniform1 \
  --dataset "PFR n40 Oracle tiny" \
  --data-type oracle \
  --target pain \
  --bias oracle_little_bias \
  --penalty "uniform @1" \
  --max-penalty "none" \
  --notes "Test, ob Beispiel-Penalties allein UNSAT vermeiden"

"""

FASTLAS_DEFAULT = "./FastLAS/FastLAS2/implementation/build/FastLAS"
RESULTS_DIR = Path("results")
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.csv"


LOG_COLUMNS = [
    "experiment_id",
    "name",
    "task",
    "dataset",
    "data_type",
    "target",
    "bias",
    "penalty",
    "max_penalty",
    "start_time",
    "end_time",
    "runtime_seconds",
    "exit_code",
    "status",
    "rule_count",
    "output_file",
    "heartbeat_file",
    "crash_file",
    "notes",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastLAS-Experiment ausführen und automatisch protokollieren."
    )

    parser.add_argument("--task", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--data-type",
        required=True,
        choices=["oracle", "cnn", "other"],
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["pain", "disgust"],
    )
    parser.add_argument("--bias", required=True)
    parser.add_argument("--penalty", default="not specified")
    parser.add_argument("--max-penalty", default="not specified")
    parser.add_argument("--notes", default="")
    parser.add_argument("--fastlas", default=FASTLAS_DEFAULT)

    return parser.parse_args()


def append_heartbeat(
    heartbeat_file: Path,
    stop_event: threading.Event,
) -> None:
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
                (
                    line.strip()
                    for line in memory.stdout.splitlines()
                    if line.startswith("Mem:")
                ),
                "memory information unavailable",
            )
        except OSError:
            memory_line = "memory information unavailable"

        with heartbeat_file.open("a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] noch aktiv | {memory_line}\n")


def determine_status(exit_code: int, output_file: Path) -> str:
    if exit_code != 0:
        return "FAILED"

    if not output_file.exists() or output_file.stat().st_size == 0:
        return "EMPTY_OUTPUT"

    output = output_file.read_text(encoding="utf-8", errors="replace")

    if "UNSATISFIABLE" in output:
        return "UNSAT"

    return "SUCCESS"


def count_rules(output_file: Path, status: str) -> int:
    if status != "SUCCESS":
        return 0

    output = output_file.read_text(encoding="utf-8", errors="replace")

    rule_pattern = re.compile(
        r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\(.*\)\s*:-",
        re.MULTILINE,
    )

    return len(rule_pattern.findall(output))


def append_experiment_log(row: dict[str, object]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    write_header = (
        not EXPERIMENT_LOG.exists()
        or EXPERIMENT_LOG.stat().st_size == 0
    )

    with EXPERIMENT_LOG.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def main() -> int:
    args = parse_arguments()

    task_path = Path(args.task)
    fastlas_path = Path(args.fastlas)

    if not task_path.is_file():
        print(f"Fehler: Task nicht gefunden: {task_path}", file=sys.stderr)
        return 2

    if not fastlas_path.is_file():
        print(
            f"Fehler: FastLAS nicht gefunden: {fastlas_path}",
            file=sys.stderr,
        )
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{timestamp}_{args.name}"

    output_file = RESULTS_DIR / f"{experiment_id}_rules.txt"
    heartbeat_file = RESULTS_DIR / f"{experiment_id}_heartbeat.log"
    crash_file = RESULTS_DIR / f"{experiment_id}_crash.log"

    start_datetime = datetime.now().astimezone()
    start_timestamp = time.monotonic()

    with heartbeat_file.open("w", encoding="utf-8") as file:
        file.write(f"Experiment-ID: {experiment_id}\n")
        file.write(f"Start: {start_datetime.isoformat(timespec='seconds')}\n")
        file.write(f"Task: {task_path}\n")
        file.write(f"PID Python: {os.getpid()}\n")

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=append_heartbeat,
        args=(heartbeat_file, stop_event),
        daemon=True,
    )
    heartbeat_thread.start()

    process: subprocess.Popen | None = None
    received_signal: str | None = None

    def handle_signal(signum: int, frame: object) -> None:
        nonlocal received_signal

        received_signal = signal.Signals(signum).name

        with crash_file.open("a", encoding="utf-8") as file:
            file.write(
                f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                f"Signal empfangen: {received_signal}\n"
            )

        if process is not None and process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGHUP, handle_signal)

    print(f"Starte Experiment: {args.name}")
    print(f"Task: {task_path}")
    print(f"Output: {output_file}")

    try:
        with output_file.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [
                    str(fastlas_path),
                    "--nopl",
                    str(task_path),
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
            )

            with heartbeat_file.open("a", encoding="utf-8") as file:
                file.write(f"PID FastLAS: {process.pid}\n")

            exit_code = process.wait()

    except OSError as error:
        exit_code = 127

        with crash_file.open("a", encoding="utf-8") as file:
            file.write(f"Startfehler: {error}\n")

    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2)

    end_datetime = datetime.now().astimezone()
    runtime_seconds = round(time.monotonic() - start_timestamp, 2)

    status = determine_status(exit_code, output_file)

    if received_signal is not None:
        status = "INTERRUPTED"

    rule_count = count_rules(output_file, status)

    with heartbeat_file.open("a", encoding="utf-8") as file:
        file.write(f"Ende: {end_datetime.isoformat(timespec='seconds')}\n")
        file.write(f"Laufzeit Sekunden: {runtime_seconds}\n")
        file.write(f"Exit-Code: {exit_code}\n")
        file.write(f"Status: {status}\n")
        file.write(f"Regelanzahl: {rule_count}\n")

    append_experiment_log(
        {
            "experiment_id": experiment_id,
            "name": args.name,
            "task": str(task_path),
            "dataset": args.dataset,
            "data_type": args.data_type,
            "target": args.target,
            "bias": args.bias,
            "penalty": args.penalty,
            "max_penalty": args.max_penalty,
            "start_time": start_datetime.isoformat(timespec="seconds"),
            "end_time": end_datetime.isoformat(timespec="seconds"),
            "runtime_seconds": runtime_seconds,
            "exit_code": exit_code,
            "status": status,
            "rule_count": rule_count,
            "output_file": str(output_file),
            "heartbeat_file": str(heartbeat_file),
            "crash_file": str(crash_file) if crash_file.exists() else "",
            "notes": args.notes,
        }
    )

    print()
    print(f"Status: {status}")
    print(f"Laufzeit: {runtime_seconds} Sekunden")
    print(f"Regeln erkannt: {rule_count}")
    print(f"Protokoll: {EXPERIMENT_LOG}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())