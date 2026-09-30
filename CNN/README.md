# CNN: action-unit-erkennung

## ziel

InceptionResnetV1 (FaceNet, pretrained auf VGGFace2) als backbone, backbone
eingefroren bis auf die letzten blöcke, eigener output-layer für 6 AUs
(AU4, AU6, AU9, AU17, AU25, AU26) x 5 thresholds (>0 bis >4). training auf
DISFA+, danach export der AU-vorhersagen auf die PFR-frames.

## wichtigste skripte

- `train.py`: training, split pro proband (siehe `config.py`), oversampling
  seltener AU-positiver frames, checkpoints nach `checkpoints/`, tensorboard
  nach `runs/`
- `evaluate.py`: auswertung pro AU und threshold (accuracy, precision,
  recall, f1, roc-auc, average precision)
- `evaluate_lopo.py`: LOPO-auswertung, ein checkpoint pro test-proband nötig
- `cnnexport_pfr_cropped_frames.py`: PFR-frames durch das CNN, ergebnis als
  CSV (AU-wahrscheinlichkeiten + binäre vorhersagen + metadaten). die CSV
  wird in `ILP/scripts/create_pfr_facts.py` zu prolog-fakten umgewandelt
- `cnnexport_predictions_disfa.py`: dasselbe für DISFA+
- `analyze_class_distribution.py`, `analyze_confusion_matrix.py`,
  `analyze_predictions.py`: label-verteilung, konfusionsmatrix, verteilung
  der vorhergesagten AUs
- `config.py` / `constants.py`: pfade, `TARGET_AUS`, thresholds,
  hyperparameter, train/val/test-probanden

## aufruf

```
cd CNN
python3 train.py
python3 evaluate.py
python3 cnnexport_pfr_cropped_frames.py
```

## io

- input: DISFA+ unter `/path/to/data/disfa_plus/{Images,Labels}` (pfade
  hartcodiert in `config.py`); PFR-frames unter `/path/to/data/relevant_pfr`
- output: `checkpoints/`, `runs/`, `outputs/predictions/…`,
  `evaluation_results/`
- die checkpoint-pfade in den `cnnexport_*.py` sind hartcodiert, vor dem lauf
  auf den gewünschten checkpoint setzen
