import torch
import random
import numpy as np
import torch.optim as optim

SEED = 42

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)


def eval(model, loader, criterion, device):

    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():

        for data, target in loader:

            data = data.to(device)
            target = target.to(device)

            outputs = model(data)

            loss = criterion(outputs, target)
            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    acc = 100.0 * correct / total
    avg_loss = running_loss / len(loader)

    return acc, avg_loss


def build_optimizer(model, optimizer_name, lr):

    name = optimizer_name.lower()

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)

    elif name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )

    elif name == "nesterov":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True
        )

    elif name == "adagrad":
        return optim.Adagrad(
            model.parameters(),
            lr=lr
        )

    elif name == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )

    elif name == "nadam":
        return optim.NAdam(
            model.parameters(),
            lr=lr
        )

    else:
        raise NotImplementedError(
            f"Optimizer {name} not implemented"
        )