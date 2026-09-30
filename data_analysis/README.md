# data_analysis: PFR-datensatz auswerten

## ziel

den PFR-n40-datensatz auswerten (annotationen, action units, demografie) und
die manuell annotierten AUs als oracle-fakten für ILP / DeepProbLog
exportieren.

## skripte

- `create_metadata.py`: legt fest, welche videos relevant sind (frontale
  druckschmerz- und frontale bildekel-videos), verknüpft die kodierten
  video-IDs mit den timeline-CSVs. ergebnis:
  `outputs/pfr_n40_pressure_pain_vs_disgust_frontal.csv`
- `analyze_annotations.py`: analyse der manuellen AU-timeline-annotationen
  (klassen-counts, AU-statistik, intensitäten, kookkurrenz, dauer,
  kombinationen), ausgabe nach `outputs/annotation_analysis/`
- `au_diagnostics.py`: zählt pro AU kurze rückfälle auf intensität 0 zwischen
  zwei aktivierungen (hinweis auf kohärenzverlust durch die randomisierte
  frame-annotation), ausgabe `flicker_summary.csv`
- `export_oracle_facts.py`: exportiert die manuell annotierten AUs als
  oracle-fakten plus beispiele für FastLAS, syntax wie in
  `ILP/task_pain_combined.las` (wg alter syntax-fehler). übernimmt den
  bestehenden train/test-split aus `ILP/data/pfr_facts_{train,test}.pl`

## aufruf

```
cd data_analysis
python3 create_metadata.py
python3 analyze_annotations.py
python3 export_oracle_facts.py --help
```

## io

- input: PFR-datensatz unter
  `/path/to/PFR_dataset` (vpcodes,
  kodierungsschlüssel, `au_annotations_n40/`, ratings)
- output: `data_analysis/outputs/`, plus für die oracle-fakten
  `ILP/oracle/data/` und `ILP/oracle/examples/`
