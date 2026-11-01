import torch
import torch.nn as nn

from model import *
from data_loader import *
from utils import *

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Loss function
criterion = nn.CrossEntropyLoss()

acc, loss = eval(
    model,
    test_loader,
    criterion,
    device
)

# Load model
model = vgg(
    cfg_vgg6,
    num_classes=10,
    batch_norm=True
).to(device)

# Load checkpoint
model.load_state_dict(
    torch.load(
        "checkpoints/best.pt",
        map_location=device
    )
)

model.eval()

# Load test data
_, test_loader = GetCifar10(64)

# Evaluate
correct = 0
total = 0
running_loss = 0.0

with torch.no_grad():

    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        outputs = model(data)

        loss = criterion(outputs, target)
        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += target.size(0)
        correct += predicted.eq(target).sum().item()

test_accuracy = 100.0 * correct / total
test_loss = running_loss / len(test_loader)

print("=" * 50)
print(f"Test Accuracy : {test_accuracy:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")
print("=" * 50)

