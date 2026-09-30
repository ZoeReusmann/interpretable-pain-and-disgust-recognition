# Interpretable Pain and Disgust Recognition

Code from my master's thesis on distinguishing **pain** from **disgust** in
facial expressions in an interpretable way. Both emotions share many facial
muscle movements, which makes them hard to separate. Instead of a black-box
classifier, this project combines a neural network for facial
**Action Unit (AU)** detection with **symbolic rule learning** and
**neuro-symbolic reasoning**, so that every decision can be traced back to
human-readable rules over facial movements.

## Approach

```
  DISFA+ --> [ CNN ] --AU predictions on PFR frames--+--> [ ILP / FastLAS ]       --> learned rules
                                                     |                                + evaluation
                                                     +--> [ DeepProbLog a/b/c ]   --> learned rule weights
                                                                                      + evaluation (incl. LOPO)

  PFR  --> [ data_analysis ] --oracle facts (manually annotated AUs)--> [ ILP / FastLAS ]
                             --video labels / metadata---------------> [ DeepProbLog a/b/c ]
```

1. **AU detection (CNN):** an InceptionResnetV1 backbone (FaceNet, pretrained
   on VGGFace2) is fine-tuned on DISFA+ to predict the intensity of six pain-
   and disgust-related AUs (AU4, AU6, AU9, AU17, AU25, AU26). The model is then
   applied to video frames of the PFR dataset.
2. **Symbolic rule learning (ILP):** frame-level AU predictions are turned into
   temporal Prolog facts (e.g. `duration_long`, `overlap`, `before`,
   `coactive`). FastLAS learns rules such as *"pain if AU25 is active for a
   long time"* from positive and negative examples.
3. **Neuro-symbolic model (DeepProbLog):** AU probabilities are aggregated to
   video level and used as probabilistic facts. DeepProbLog learns the weights
   of three rule variants: literature-based (a), ILP-learned (b) and a reduced
   variant using the two most discriminative features (c).
4. **Evaluation:** subject-wise train/test splits and leave-one-person-out
   (LOPO) cross-validation, standard metrics (accuracy, precision, recall, F1,
   MCC) and rule-stability analysis across folds.

To tell whether errors come from AU detection or from the rules, the ILP part
is also run on **oracle facts**: the manually annotated AUs of the dataset.

## Repository structure

| Folder | Content |
| --- | --- |
| `CNN/` | AU model: training, evaluation, export of AU predictions |
| `ILP/` | fact and example generation, FastLAS tasks, background knowledge, rule evaluation |
| `DeepProbLog/` | AU aggregation, rule models a/b/c, training and LOPO evaluation |
| `data_analysis/` | dataset analysis, metadata, export of oracle facts |

Each folder has its own README with details (in German).

## Tech stack

Python, PyTorch, facenet-pytorch, OpenCV, pandas, scikit-learn, SciPy,
matplotlib/seaborn, TensorBoard, Prolog / ASP, FastLAS, DeepProbLog / ProbLog.

## Setup

```
pip install -r requirements.txt
pip install deepproblog==2.0.6 problog==2.2.9   # only for DeepProbLog / ILP evaluation
```

FastLAS is an external binary (https://github.com/spike-imperial/FastLAS),
expected under `ILP/FastLAS/`. See `ILP/README.md`.

## Pipeline order

1. `data_analysis/`: annotations and metadata, then `export_oracle_facts.py`
2. `CNN/`: training, then `cnnexport_*.py` for the AU predictions
3. `ILP/`: `scripts/create_*.py`, then `run.sh` / `protocol.py`
4. `DeepProbLog/`: `preprocessing/`, `build_programs.py`, then `protocol.py` /
   `run_lopo_training.py`

## Data

This repository contains code, background knowledge, language bias and rule
definitions only. **No datasets, derived facts, trained weights or results are
included.** DISFA+ and PFR have to be obtained separately from their
providers. Paths in the code are placeholders (`/path/to/data`,
`/path/to/PFR_dataset`); for `CNN/config.py` the environment variable
`DATA_DIR` is sufficient.

## License

MIT, see [LICENSE](LICENSE).
