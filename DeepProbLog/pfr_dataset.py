import csv

from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Term

"""
Dataset-Wrapper analog zu AdditionDataset (MNIST-Beispiel).
Liest video_labels.csv (video_atom, label) und erzeugt pro Video 2 Queries: eine für pain(V), eine für disgust(V), jeweils mit Ziel-
wahrscheinlichkeit p=1.0 (falls das Label zutrifft) oder p=0.0 (falls nicht).

Preprocessing Schritt, deswegen...
Kein Netzwerk nötig: alle AU-Wahrscheinlichkeiten sind bereits als feste probabilistische Fakten im .pl-Programm enthalten (CNN ist eingefroren und wurde offline ausgewertet).
"""
class PFRDataset(Dataset):
    def __init__(self, labels_csv, video_atoms_subset=None):
        """
        :param labels_csv: Pfad zu video_labels.csv (video_atom,label)
        :param video_atoms_subset: optionale Liste/Set von video_atoms,
            um Train/Test-Splits zu realisieren. Falls None, werden alle
            Videos aus der CSV verwendet.
        """
        self.entries = []  # Liste von (video_atom, label)
        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row["video_atom"]
                label = row["label"]
                if video_atoms_subset is not None and vid not in video_atoms_subset:
                    continue
                self.entries.append((vid, label))

        # Pro Video zwei Queries (pain, disgust) -> Liste linearisieren
        self.queries = []
        for vid, label in self.entries:
            # WICHTIG: in video_labels.csv steht "pressure_pain", nicht "pain"
            # (siehe pfr_n40_pressure_pain_vs_disgust_frontal.csv, Spalte 'label')
            is_pain = label == "pressure_pain"
            is_disgust = label == "disgust"
            self.queries.append(
                Query(Term("pain", Term(vid)), p=1.0 if is_pain else 0.0)
            )
            self.queries.append(
                Query(Term("disgust", Term(vid)), p=1.0 if is_disgust else 0.0)
            )

    def __len__(self):
        return len(self.queries)

    def to_query(self, i):
        return self.queries[i]