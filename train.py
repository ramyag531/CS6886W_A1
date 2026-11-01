import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model import *
from data_loader import *
from utils import *
import wandb

wandb.login()

train_loader,test_loader = GetCifar10(64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = vgg(cfg_vgg6, num_classes=10, batch_norm=True).to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.0005
epochs = 10
optimizer = build_optimizer(model, "adam", lr)

train_losses = []
train_accs = []
test_accs = []
val_losses = []
val_accs = []

wandb.init(
    project="vgg6-cifar10",
    config={
        "activation": "silu",
        "optimizer": "adam",
        "learning_rate": lr,
        "batch_size": 64,
        "epochs": epochs,
        "batch_norm": True
    }
)

def train_model(model, epochs, optimizer, train_loader, test_loader):

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        for data, target in train_loader:

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            outputs = model(data)

            loss = criterion(outputs, target)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        # Training loss for current epoch
        train_loss = running_loss / len(train_loader)

        # Training metrics
        train_acc, train_eval_loss = eval(model, train_loader)

        # Validation metrics
        val_acc, val_loss = eval(model, test_loader)

        # Store values for plots
        train_losses.append(train_loss)

        train_accs.append(train_acc)

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc
        })

        print(f"Epoch {epoch} - Train_Loss: {running_loss/len(train_loader):.4f} , Train_acc: {train_acc}, Test_acc : {val_acc}")
        
    wandb.finish()

torch.save(
    model.state_dict(),
    "checkpoints/best.pt"
)

