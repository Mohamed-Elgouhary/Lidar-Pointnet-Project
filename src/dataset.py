import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.expanduser("~/lidar_pointnet_project")
DATA_DIR = os.path.join(BASE_DIR, "data/processed/normalized_1024")
METADATA_CSV = os.path.join(BASE_DIR, "data/metadata/samples_metadata.csv")


class KITTIPointCloudDataset(Dataset):
    def __init__(self, split="train"):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.samples = self._load_metadata()

    def _load_metadata(self):
        samples = []
        with open(METADATA_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != self.split:
                    continue

                row["label"] = int(row["label"])
                row["object_index"] = int(row["object_index"])
                row["num_points"] = int(row["num_points"])
                samples.append(row)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        file_name = sample_info["file_name"]
        file_path = os.path.join(DATA_DIR, file_name)

        data = np.load(file_path, allow_pickle=True)

        points = data["points"].astype(np.float32)   # (1024, 3)
        label = int(data["label"])

        points_tensor = torch.tensor(points, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return points_tensor, label_tensor


def inspect_dataset(split="train", batch_size=8):
    print("=" * 70)
    print(f"Inspecting dataset split = {split}")
    print("=" * 70)

    dataset = KITTIPointCloudDataset(split=split)
    print(f"Number of samples in {split}: {len(dataset)}")

    points, label = dataset[0]
    print("\nSingle sample:")
    print(f"  points shape : {points.shape}")
    print(f"  points dtype : {points.dtype}")
    print(f"  label        : {label}")
    print(f"  label dtype  : {label.dtype}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    batch_points, batch_labels = next(iter(loader))

    print("\nOne batch from DataLoader:")
    print(f"  batch points shape : {batch_points.shape}")
    print(f"  batch labels shape : {batch_labels.shape}")
    print(f"  batch points dtype : {batch_points.dtype}")
    print(f"  batch labels dtype : {batch_labels.dtype}")

    print("\nFirst few labels in batch:")
    print(batch_labels[:10])

    print("\nStep 11 passed: Dataset and DataLoader work correctly.")


if __name__ == "__main__":
    inspect_dataset(split="train", batch_size=8)
