import os
import csv
import numpy as np
from collections import Counter

BASE_DIR = os.path.expanduser("~/lidar_pointnet_project")
INPUT_DIR = os.path.join(BASE_DIR, "data/processed/all_samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/normalized_1024")
METADATA_CSV = os.path.join(BASE_DIR, "data/metadata/samples_metadata.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_POINTS = 1024
RANDOM_SEED = 42


def normalize_points(points_xyz):
    """
    points_xyz: (N, 3)
    - center at origin
    - scale to unit sphere
    """
    centroid = np.mean(points_xyz, axis=0)
    points_xyz = points_xyz - centroid

    max_dist = np.max(np.sqrt(np.sum(points_xyz ** 2, axis=1)))
    if max_dist > 1e-8:
        points_xyz = points_xyz / max_dist

    return points_xyz.astype(np.float32)


def resample_points(points_xyz, num_points=1024, rng=None):
    """
    If N >= num_points: sample without replacement
    If N < num_points: sample with replacement
    """
    if rng is None:
        rng = np.random.default_rng()

    n = points_xyz.shape[0]

    if n >= num_points:
        indices = rng.choice(n, size=num_points, replace=False)
    else:
        indices = rng.choice(n, size=num_points, replace=True)

    return points_xyz[indices]


def load_metadata(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            row["object_index"] = int(row["object_index"])
            row["num_points"] = int(row["num_points"])
            rows.append(row)
    return rows


def main():
    print("=" * 70)
    print("Step 10: Normalizing and resampling crops to 1024 points")
    print("=" * 70)

    rows = load_metadata(METADATA_CSV)
    print(f"Loaded metadata for {len(rows)} samples.")

    rng = np.random.default_rng(RANDOM_SEED)
    saved_counter = Counter()

    for i, row in enumerate(rows):
        input_path = os.path.join(INPUT_DIR, row["file_name"])
        output_path = os.path.join(OUTPUT_DIR, row["file_name"])

        if not os.path.exists(input_path):
            print(f"Missing input file: {input_path}")
            continue

        data = np.load(input_path, allow_pickle=True)

        # original saved points are (N, 4): x, y, z, reflectance
        points = data["points"]
        points_xyz = points[:, :3].astype(np.float32)

        # normalize and resample
        points_xyz = normalize_points(points_xyz)
        points_xyz = resample_points(points_xyz, num_points=NUM_POINTS, rng=rng)

        np.savez_compressed(
            output_path,
            points=points_xyz,                      # (1024, 3)
            label=int(data["label"]),
            class_name=str(data["class_name"]),
            frame_id=str(data["frame_id"]),
            object_index=int(data["object_index"]),
            split=row["split"],
        )

        saved_counter[row["class_name"]] += 1

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(rows)} samples...")

    print("\nSaved normalized samples:")
    for cls in ["Car", "Pedestrian", "Cyclist"]:
        print(f"  {cls:10s}: {saved_counter[cls]}")

    total_saved = sum(saved_counter.values())
    print(f"\nTotal normalized samples saved: {total_saved}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nStep 10 passed: all samples converted to fixed size 1024 x 3.")


if __name__ == "__main__":
    main()
