CS6886W – Assignment 1: VGG6 Modular Implementation

This repository contains a fully modular PyTorch implementation of a VGG6-style CNN trained on the CIFAR-10 dataset.
It includes code for training, testing, Weights & Biases sweeps, and final model checkpoints — satisfying all deliverables for CS6886W Assignment 1.
📁 Project Structure
File 	Description
model.py 	Defines the VGG6 CNN architecture (activation-configurable, with BatchNorm).
data_loader.py 	Prepares CIFAR-10 dataset loaders with normalization, augmentation + Cutout.
train.py 	Training script for a single configuration; saves latest.pt and best.pt checkpoints.
test.py 	Loads a trained checkpoint and evaluates on the CIFAR-10 test set.
utils.py 	Helper utilities (accuracy, training loop, optimizer/scheduler setup, seeding).
sweep_full.py 	Complete 25-run W&B sweep pipeline + charts + final retrain and test.
checkpoints/ 	Contains trained model weights (best.pt, latest.pt, etc.) tracked with Git LFS.
README.md 	This documentation and grader verification steps.
⚙️ Best Configuration (Highest Validation Accuracy)
Hyper-parameter 	Value
Activation 	gelu
Optimizer 	rmsprop
Learning Rate 	5 × 10⁻⁴
Batch Size 	64
Epochs 	10
Weight Init 	Kaiming
Batch Norm 	Enabled
Data Augmentation 	Enabled (Crop + Flip + Cutout)
seed Config 	seed=42

Best trained model checkpoint:
🧪 How to Verify the Trained Model

Follow these steps to evaluate the trained model directly.
1 Clone the repository

git clone https://github.com/ramyag531/CS6886W_A1.git
cd CS6886W_A1

2 Install dependencies

pip install torch torchvision matplotlib plotly pandas wandb

3 Training the Model (Optional) or else use the best.pt saved in checkpoint

python train.py --activation gelu --optimizer rmsprop --lr 5e-4 --batch-size 64 --epochs 10 --augment --batch-norm

4 Verifying the Model

python test.py --ckpt checkpoints/best.pt --activation gelu --batch-norm


Simple steps:
git clone https://github.com/ramyag531/CS6886W_A1.git
cd CS6886W_A1
pip install torch torchvision
python test.py --ckpt checkpoints/best.pt --activation gelu --batch-norm

