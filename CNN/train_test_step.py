import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import get_dataloaders
from model import AUModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, _ = get_dataloaders()

model = AUModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

images, raw_labels, threshold_targets = next(iter(train_loader))

images = images.to(device)
threshold_targets = threshold_targets.to(device)

outputs = model(images)
loss = criterion(outputs, threshold_targets)

print("images shape:", images.shape)
print("outputs shape:", outputs.shape)
print("targets shape:", threshold_targets.shape)
print("loss:", loss.item())

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("ein trainingsschritt erfolgreich")