import os
import numpy as np
from collections import Counter

BASE_DIR = os.path.expanduser("~/lidar_pointnet_project")
VELODYNE_DIR = os.path.join(BASE_DIR, "data/kitti_raw/training/velodyne")
LABEL_DIR = os.path.join(BASE_DIR, "data/kitti_raw/training/label_2")
CALIB_DIR = os.path.join(BASE_DIR, "data/kitti_raw/training/calib")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/all_samples")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
MIN_POINTS = 30

# Cap the number of saved samples per class
MAX_SAMPLES_PER_CLASS = {
    "Car": 1200,
    "Pedestrian": 1200,
    "Cyclist": 1200,
}


def read_point_cloud(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def read_label_file(label_path):
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            obj = {
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": np.array(list(map(float, parts[4:8])), dtype=np.float32),
                "h": float(parts[8]),
                "w": float(parts[9]),
                "l": float(parts[10]),
                "x": float(parts[11]),
                "y": float(parts[12]),
                "z": float(parts[13]),
                "rotation_y": float(parts[14]),
            }
            objects.append(obj)
    return objects


def read_calib_file(calib_path):
    calib = {}
    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            calib[key] = np.array([float(x) for x in value.strip().split()], dtype=np.float32)
    return calib


def build_transformation_matrices(calib):
    Tr_velo_to_cam = calib["Tr_velo_to_cam"].reshape(3, 4)
    Tr_velo_to_cam_4x4 = np.eye(4, dtype=np.float32)
    Tr_velo_to_cam_4x4[:3, :4] = Tr_velo_to_cam

    R0_rect = calib["R0_rect"].reshape(3, 3)
    R0_rect_4x4 = np.eye(4, dtype=np.float32)
    R0_rect_4x4[:3, :3] = R0_rect

    return Tr_velo_to_cam_4x4, R0_rect_4x4


def transform_velodyne_to_rectified_camera(points_xyz, Tr_velo_to_cam, R0_rect):
    N = points_xyz.shape[0]
    points_h = np.hstack([points_xyz, np.ones((N, 1), dtype=np.float32)])
    points_cam = (R0_rect @ (Tr_velo_to_cam @ points_h.T)).T
    return points_cam[:, :3]


def points_in_box(points_rect, obj):
    h, w, l = obj["h"], obj["w"], obj["l"]
    x, y, z = obj["x"], obj["y"], obj["z"]
    ry = obj["rotation_y"]

    box_center = np.array([x, y - h / 2.0, z], dtype=np.float32)
    pts = points_rect - box_center

    c = np.cos(ry)
    s = np.sin(ry)
    R = np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c]
    ], dtype=np.float32)

    pts_local = pts @ R

    mask = (
        (pts_local[:, 0] >= -l / 2.0) & (pts_local[:, 0] <=  l / 2.0) &
        (pts_local[:, 1] >= -h / 2.0) & (pts_local[:, 1] <=  h / 2.0) &
        (pts_local[:, 2] >= -w / 2.0) & (pts_local[:, 2] <=  w / 2.0)
    )
    return mask


def get_common_frame_ids():
    velodyne_ids = {f.replace(".bin", "") for f in os.listdir(VELODYNE_DIR) if f.endswith(".bin")}
    label_ids = {f.replace(".txt", "") for f in os.listdir(LABEL_DIR) if f.endswith(".txt")}
    calib_ids = {f.replace(".txt", "") for f in os.listdir(CALIB_DIR) if f.endswith(".txt")}
    return sorted(list(velodyne_ids & label_ids & calib_ids))


def class_caps_reached(saved_counter):
    for cls, cap in MAX_SAMPLES_PER_CLASS.items():
        if saved_counter[cls] < cap:
            return False
    return True


def main():
    frame_ids = get_common_frame_ids()
    saved_counter = Counter()
    skipped_small_counter = Counter()
    examined_counter = Counter()
    total_saved = 0

    print("=" * 70)
    print("Step 8: Building processed point-cloud classification dataset")
    print("=" * 70)
    print(f"Total common KITTI frames available: {len(frame_ids)}")
    print(f"Minimum points per crop: {MIN_POINTS}")
    print(f"Class caps: {dict(MAX_SAMPLES_PER_CLASS)}")
    print(f"Output directory: {OUTPUT_DIR}")

    for frame_idx, frame_id in enumerate(frame_ids):
        if class_caps_reached(saved_counter):
            print("\nReached all class caps. Stopping early.")
            break

        bin_path = os.path.join(VELODYNE_DIR, frame_id + ".bin")
        label_path = os.path.join(LABEL_DIR, frame_id + ".txt")
        calib_path = os.path.join(CALIB_DIR, frame_id + ".txt")

        points = read_point_cloud(bin_path)
        objects = read_label_file(label_path)
        calib = read_calib_file(calib_path)

        Tr_velo_to_cam, R0_rect = build_transformation_matrices(calib)
        points_xyz = points[:, :3]
        points_rect = transform_velodyne_to_rectified_camera(points_xyz, Tr_velo_to_cam, R0_rect)

        for obj_idx, obj in enumerate(objects):
            cls = obj["type"]

            if cls not in TARGET_CLASSES:
                continue

            examined_counter[cls] += 1

            if saved_counter[cls] >= MAX_SAMPLES_PER_CLASS[cls]:
                continue

            mask = points_in_box(points_rect, obj)
            crop_points = points[mask]

            if crop_points.shape[0] < MIN_POINTS:
                skipped_small_counter[cls] += 1
                continue

            save_name = f"{frame_id}_{obj_idx:03d}_{cls}.npz"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            np.savez_compressed(
                save_path,
                points=crop_points,
                label=TARGET_CLASSES[cls],
                class_name=cls,
                frame_id=frame_id,
                object_index=obj_idx
            )

            saved_counter[cls] += 1
            total_saved += 1

        if (frame_idx + 1) % 250 == 0:
            print(f"\nProcessed {frame_idx + 1} frames...")
            print(f"Saved so far: {dict(saved_counter)}")
            print(f"Skipped small so far: {dict(skipped_small_counter)}")

    print("\n" + "=" * 70)
    print("Step 8 finished")
    print("=" * 70)

    print("\nExamined target objects:")
    for cls in ["Car", "Pedestrian", "Cyclist"]:
        print(f"  {cls:10s}: {examined_counter[cls]}")

    print("\nSaved samples:")
    for cls in ["Car", "Pedestrian", "Cyclist"]:
        print(f"  {cls:10s}: {saved_counter[cls]}")

    print("\nSkipped because too few points:")
    for cls in ["Car", "Pedestrian", "Cyclist"]:
        print(f"  {cls:10s}: {skipped_small_counter[cls]}")

    print(f"\nTotal saved samples: {total_saved}")
    print(f"Files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
