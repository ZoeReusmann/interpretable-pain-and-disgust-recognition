# DeepProbLog: neuro-symbolisches modell a/b/c

## ziel

das CNN ist eingefroren, die AU-wahrscheinlichkeiten sind offline zu
video-ebene aggregiert und stehen als feste probabilistische fakten im
programm. DeepProbLog lernt nur die regelgewichte (`t(_)`-parameter). drei
regelvarianten a/b/c, jeweils auch LOPO.

- modell a: literaturbasierte regelstruktur (kunz et al., prkachin, ekman &
  friesen), vor allem über `duration_long`
- modell b: die aus dem ILP-teil übernommene, FastLAS-gelernte regelstruktur
  (aus `ILP/`), nutzt `overlap` und `duration_long`
- modell c: die reduzierte variante, nur die zwei merkmale mit größter
  gemessener klassentrennung, je einzeln ohne UND-verknüpfung

## setup

```
pip install deepproblog==2.0.6 problog==2.2.9
```

der DeepProbLog-quellcode plus die beispiele (`deepproblog_examples/`) sind
nicht im repo.

## ablauf

alle skripte werden aus dem repo-root aufgerufen (pfade sind darauf relativ).

```
python3 DeepProbLog/preprocessing/aggregate_au_probs.py
python3 DeepProbLog/preprocessing/make_split.py
python3 DeepProbLog/build_programs.py
python3 DeepProbLog/protocol.py --model model_a --name lauf1 --dataset "PFR n40"
python3 DeepProbLog/run_lopo_training.py
python3 DeepProbLog/evaluation/evaluate_lopo.py
```

## dateien

- `preprocessing/aggregate_au_probs.py`: frame-AU-wahrscheinlichkeiten aus dem
  CNN-export zu video-ebene fakten (`duration_long`, `overlap`, `presence`)
  plus query-datei mit den pain / disgust-labels
- `preprocessing/make_split.py`: subject-wise train / test-split
- `build_programs.py`: baut `model_{a,b,c}_full.pl` aus
  `facts/au_facts.pl` + der jeweiligen `models/model_x_rules.pl`
  (zusammenkonkateniert, ProbLog kennt kein `include`)
- `models/model_{a,b,c}_rules.pl`: regeldefinitionen, `*_full.pl` = fakten +
  regeln
- `facts/`: `au_facts.pl`, `video_labels.csv`, `train_videos.txt` /
  `test_videos.txt`
- `train.py`: trainings-baustein, loggt nicht selbst
- `protocol.py`: runner mit protokoll (`results/experiment_log.csv`), wrappt
  `train.py`
- `run_lopo_training.py`: LOPO in einem python-prozess (kein GPU-wechsel
  nötig), pro fold ein gelernter zustand als eigene datei
- `lopo_checkpoints/`: gelernter zustand pro test-proband (pendant zum
  CNN-checkpoint)
- `evaluation/evaluate_lopo.py`: lädt alle LOPO-checkpoints, wertet je auf
  dem zurückgehaltenen proband aus, führt die ergebnisse zusammen
