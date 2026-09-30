import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from constants import TARGET_AUS, DEFAULT_THRESHOLD, IMAGE_SIZE


class AUDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, threshold=DEFAULT_THRESHOLD):
        """
        
        Ziel: Rückgabe pro Sample
        - image (z.B. 000.jpg)
        - au_label_raw: Tensor (z.B: [0, 3, 0, 1, 2, 0]) mit Intensitäten der AUs - Rohwerte
        - au_threshold_targets: Tensor (z.B. [0, 1, 0, 0, 1, 0]) mit binären Werten basierend auf Schwelle

        Ordnerstruktur Beispiel DISFA+:
        Images/SN001/A1_AU1_TrailNo_1/028.jpg
        Labels/SN001/A1_AU1_TrailNo_1/AU4.txt

        Parameter:
        au_labels_raw: zeigen INtensitäten 0-5
        au_threshold_targets: binäre Targets (0/1) basierend auf Schwelle
        image_dir: Pfad zu Bildern des DISFA+ Datasets
        label_dir: Pfad zu AU Label Dateien des DISFA+ Datasets
        transform: torchvision transforms
        threshold: Schwelle für binäre Targets (z.B. >=2 → aktiv)

        Reihenfolge der Indizes:
        index 0 = AU4
        index 1 = AU6
        index 2 = AU9
        index 3 = AU17
        index 4 = AU25
        index 5 = AU26

        Plus prüft, ob Image Pfade stimmen, ergänzung nach erstem Training
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.threshold = threshold

        self.target_aus = TARGET_AUS
        self.au_to_idx = {au: idx for idx, au in enumerate(self.target_aus)}
        
        # Check: Pfad zu Label und Image Ordner gefunden?
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image Pfad nicht gefunden: {self.image_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label Pfad nicht gefunden: {self.label_dir}")
        
        self.samples = self._load_samples()

    def _is_valid_image(self, img_path):
        if not os.path.isfile(img_path):
            return False

        if os.path.getsize(img_path) == 0:
            return False

        try:
            with Image.open(img_path) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False

    def _load_samples(self):
        samples = []

        for subject in sorted(os.listdir(self.image_dir)):
            subject_img_path = os.path.join(self.image_dir, subject)

            if not os.path.isdir(subject_img_path):
                continue

            if subject.startswith('.'):
                continue
        
            for trial in sorted(os.listdir(subject_img_path)):
                trial_img_path = os.path.join(subject_img_path, trial)

                if not os.path.isdir(trial_img_path):
                    continue

                if trial.startswith('.'):
                    continue

                # falls es passende Labels für das Trial gibt
                trial_label_path = os.path.join(self.label_dir, subject, trial)
                if not os.path.isdir(trial_label_path):
                    continue

                for img_file in sorted(os.listdir(trial_img_path)):
                    if img_file.startswith('.'):
                        continue
                    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    img_path = os.path.join(trial_img_path, img_file)
                    frame_name = img_file

                    if not self._is_valid_image(img_path):
                        print(f"Überspringe ungültiges Bild: {img_path}")
                        continue

                    au_labels_raw = [] # Liste für ein Frame mit Intensitäten

                    for au in TARGET_AUS: # nur die target AUs durchgehen, target AU's sind alle die ich tracke
                        label_file = os.path.join(trial_label_path, f"{au}.txt")

                        if not os.path.isfile(label_file): # passende Labels laden
                            continue

                        with open(label_file, "r") as f: #r:lesemodus 
                            for line in f:
                                fname, intensity = line.strip().split()
                                if fname == frame_name:
                                    au_labels_raw.append(int(intensity))
                                    break
                    
                    if not any(label > 0 for label in au_labels_raw): # eines der AU4, AU6, AU9, AU17, AU25, AU26 > 0 ->Bild mitnehmen
                        continue  # überspringe Frames ohne aktive AU


                    samples.append({
                        "image_path": img_path,
                        "subject": subject,
                        "trial": trial,
                        "frame": frame_name
                    })

        return samples

    def _load_labels(self, au_file_path):
        """
        Ziel: Label Dateien einlesen und frames zuordnen
         Input: z.B. 000.jpg 0
         ersten 2 Zeilen der Label .txt Kommentare
         Rückgabe: dict[frame_name] = intensität
        """
        label_dict = {}

        with open(au_file_path, "r") as f:
            for line in f:
                line = line.strip()
                # Kommentare am Anfang der Datei überspringen
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) != 2:
                    continue

                frame_name, value = parts
                label_dict[frame_name] = float(value)

        return label_dict

    def _load_labels_for_frame(self, subject, trial, frame_name):
        """
        Ziel: pro Bild alle target AUs in richtiger Reihenfolge ausgeben
        """
        trial_label_path = os.path.join(self.label_dir, subject, trial)
        au_values = []

        for au in self.target_aus:
            au_file = f"{au}.txt"
            au_file_path = os.path.join(trial_label_path, au_file)

            # falls AU Werte oder Frame nicht gefunden wird, Vorkehrungen:
            if not os.path.exists(au_file_path):
                au_values.append(0.0)  # fehlende AU mit 0.0 auffüllen
                continue

            label_dict = self._load_labels(au_file_path)

            if frame_name not in label_dict:
                raise ValueError(
                    f"{frame_name} nicht gefunden in {au_file_path}"
                )
            au_values.append(label_dict[frame_name])

        return torch.tensor(au_values, dtype=torch.float)
    
    def _intensity_to_threshold_targets(self, intensities):
        """
        Ziel: Intensitäten in binäre Targets umwandeln
        Muster:
        0 -> [0,0,0,0,0]
        1 -> [1,0,0,0,0]
        2 -> [1,1,0,0,0]
        3 -> [1,1,1,0,0]
        4 -> [1,1,1,1,0]
        5 -> [1,1,1,1,1]
        Input: intensities (Tensor [n_aus])
        Output: threshold (Tensor [n_aus, 5])
        """
        thresholds = []

        for intensity in intensities:
            threshold_vector = [
                1.0 if intensity >= t else 0.0
                for t in range(1, 6)  # 6 = threshold + 1
            ]
            thresholds.append(threshold_vector)
        
        return torch.tensor(thresholds, dtype=torch.float)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = sample["image_path"]
        subject = sample["subject"]
        trial = sample["trial"]
        frame_name = sample["frame"]
        
        # Image
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Fehler beim Laden von Bild {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        # raw AU intensities
        au_labels_raw = self._load_labels_for_frame(subject, trial, frame_name)

        # Threshold targets
        au_threshold_targets = self._intensity_to_threshold_targets(au_labels_raw)

        metadata = { # für eine zeiltiche IEnordnung später
            "subject": subject,
            "trial": trial,
            "frame": frame_name,
            "image_path": img_path,
        }

        return image, au_labels_raw, au_threshold_targets, metadata


        # evtl. noch daran arbeiten, fehlende AU wird mit 0.0 aufgefüllt, evtl Fehler ausgeben lassen?
        # missing_aus = [au for au in self.target_aus if au not in label_dict]
        # if missing_aus:
        #     raise ValueError(f"Fehlende AUs in {label_path}: {missing_aus}")
        # TO DO nach test: 06.04 beim initialisieren einmal alle Label-Dateien in Speicherstrukturen laden, dann bei __getitem__() nurnoch nachschlagen