import torch
import torch.nn as nn
from constants import N_AUS, N_THRESHOLDS
from facenet_pytorch import InceptionResnetV1

"""
Kontext
Backbone: InceptionResnetV1 (FaceNet) pretrained auf VGGFace2, freezed, nur letzte Schicht eigener Output-Layer
Output: 6 AUs * 5 Thresholds = 30 Werte pro Bild

"""

class AUModel(nn.Module):
    def __init__(self, num_aus=N_AUS, num_threshold=N_THRESHOLDS):
        super().__init__()

        self.num_aus = num_aus
        self.num_thresholds = num_threshold

        # FaceNet / InceptionResnetV1 als Backbone
        # pretrained auf VGGFace2 statt ImageNet
        self.backbone = InceptionResnetV1(pretrained="vggface2")

        # FaceNet gibt 512 Features pro Bild aus
        in_features = 512

        # eigener Output-Layer: 6 AUs * 5 Thresholds = 30 Werte
        self.classifier = nn.Linear(in_features, num_aus * num_threshold)

    def freeze_backbone(self):
        # Erst gesamten Backbone einfrieren
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Letzte FaceNet-Blöcke wieder trainierbar machen
        for param in self.backbone.block8.parameters():
            param.requires_grad = True

        for param in self.backbone.last_linear.parameters():
            param.requires_grad = True

        for param in self.backbone.last_bn.parameters():
            param.requires_grad = True

        # Eigene Klassifikationsschicht trainierbar machen
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)          # [B, 512]
        x = self.classifier(x)        # [B, 30]
        x = x.view(-1, self.num_aus, self.num_thresholds)  # [B, 6, 5]
        return x