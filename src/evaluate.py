import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from dataset import KITTIPointCloudDataset
from model import SimplePointNet

BASE_DIR = os.path.expanduser("~/lidar_pointnet_project")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "outputs/checkpoints/best_pointnet_model.pth")
HISTORY_PATH = os.path.join(BASE_DIR, "outputs/logs/training_history.json")
FIGURE_DIR = os.path.join(BASE_DIR, "outputs/figures")

os.makedirs(FIGURE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(history, save_dir):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss curve
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("Step 15: Evaluating best model and saving figures")
    print("=" * 70)
    print(f"Using device: {DEVICE}")

    test_dataset = KITTIPointCloudDataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    model = SimplePointNet(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for points, labels in test_loader:
            points = points.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(points)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    print(f"\nTest accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        digits=4
    )
    print("\nClassification Report:")
    print(report)

    # Save confusion matrix figure
    cm_path = os.path.join(FIGURE_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)

    # Save training curves
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)

    plot_training_curves(history, FIGURE_DIR)

    print("Saved figures:")
    print(f"  Confusion matrix : {cm_path}")
    print(f"  Loss curve       : {os.path.join(FIGURE_DIR, 'loss_curve.png')}")
    print(f"  Accuracy curve   : {os.path.join(FIGURE_DIR, 'accuracy_curve.png')}")

    print("\nStep 15 passed: evaluation figures saved successfully.")


if __name__ == "__main__":
    main()
