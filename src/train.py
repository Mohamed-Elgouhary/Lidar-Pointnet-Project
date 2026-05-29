import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import KITTIPointCloudDataset
from model import SimplePointNet

BASE_DIR = os.path.expanduser("~/lidar_pointnet_project")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "outputs/checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "outputs/logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_pointnet_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_pointnet_model.pth")
HISTORY_PATH = os.path.join(LOG_DIR, "training_history.json")


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total


def run_one_epoch(model, loader, criterion, optimizer=None):
    """
    If optimizer is provided -> training mode
    Otherwise -> evaluation mode
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for points, labels in loader:
        points = points.to(DEVICE)   # (B, 1024, 3)
        labels = labels.to(DEVICE)   # (B,)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(points)               # (B, 3)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        correct, total = compute_accuracy(logits, labels)
        total_correct += correct
        total_samples += total

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def main():
    print("=" * 70)
    print("Step 13: Training SimplePointNet")
    print("=" * 70)
    print(f"Using device: {DEVICE}")

    train_dataset = KITTIPointCloudDataset(split="train")
    val_dataset = KITTIPointCloudDataset(split="val")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = SimplePointNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Saved new best model to {BEST_MODEL_PATH}")

    # Save last model
    torch.save(model.state_dict(), LAST_MODEL_PATH)

    # Save training history
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("Training finished")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Last model saved to: {LAST_MODEL_PATH}")
    print(f"Training history saved to: {HISTORY_PATH}")
    print("\nStep 13 passed: training script completed successfully.")


if __name__ == "__main__":
    main()
