# ILP: regel-lernen mit FastLAS

## ziel

aus den CNN-AU-fakten (bzw. den oracle-annotationen aus data_analysis) mit
FastLAS2 symbolische regeln für schmerz / ekel lernen und gegen einen
test-split auswerten.

## externes binary

FastLAS ist nicht teil des repos. erwartet unter
`ILP/FastLAS/FastLAS2/implementation/build/FastLAS` (siehe `run.sh` und
`protocol.py`). quelle: https://github.com/spike-imperial/FastLAS

## wichtigste dateien

- `background/background_knowledge.pl`: hintergrundwissen (zeitliche
  prädikate: before, next, coactive, overlap, starts_before, duration_long …)
- `bias/*.las`: sprach-bias (mode declarations, penalties)
- `scripts/create_pfr_facts.py`: CNN-export-CSV zu prolog-fakten (ohne label)
- `scripts/create_examples.py`: positive / negative beispiele je für pain und
  disgust als `.las`
- `scripts/create_train_test_split.py`: fakten und beispiele in train / test
  aufteilen, split pro proband
- `run.sh`: FastLAS auf einen task ansetzen, ausgabe plus heartbeat und
  crashlog nach `results/`
- `protocol.py`: runs protokollieren (`results/experiment_log.csv`), wrappt
  den FastLAS-aufruf und loggt start / ende / laufzeit / status
- `evaluation/evaluate.py`: gelernte regel gegen testdaten (accuracy,
  precision, recall, f1, mcc, TP/FP/TN/FN), rohzählungen zusätzlich als JSON
- `evaluation/compare_rule_folds.py`: regelstabilität über LOPO-folds
  (jaccard über kanonisierte body-literale)
- `learned_rules/`: ablage für die gelernten regeln (`.lp`), nicht im repo
- `oracle/`: tasks und daten, die auf den oracle-fakten aus data_analysis
  laufen statt auf den CNN-fakten

## aufruf

```
cd ILP
python3 scripts/create_pfr_facts.py
python3 scripts/create_examples.py
python3 scripts/create_train_test_split.py
./run.sh
python3 evaluation/evaluate.py --help
```

`run.sh` und `protocol.py` erwarten das FastLAS-binary am oben genannten pfad.
