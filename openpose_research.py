#!/usr/bin/env python3
"""OpenPose BODY_25 2D (4 views) -> DLT 3D reconstruction + kinematics.

Input:
- 4 view directories containing OpenPose JSONs (frame_XXXXX_keypoints.json)
- P matrices (3x4) for each view in JSON or NPY

Outputs:
- keypoints_3d.csv (x,y,z,conf per joint)
- angles.csv (joint angles in degrees)
- kinematics.csv (speed, acceleration magnitude)
- summary.csv (basic stats for angles and speeds)
"""

import argparse
import json
import math
from pathlib import Path
import csv
from typing import List, Tuple

import numpy as np


BODY_25_NAMES = [
    "nose",         # 0
    "neck",         # 1
    "r_shoulder",   # 2
    "r_elbow",      # 3
    "r_wrist",      # 4
    "l_shoulder",   # 5
    "l_elbow",      # 6
    "l_wrist",      # 7
    "mid_hip",      # 8
    "r_hip",        # 9
    "r_knee",       # 10
    "r_ankle",      # 11
    "l_hip",        # 12
    "l_knee",       # 13
    "l_ankle",      # 14
    "r_eye",        # 15
    "l_eye",        # 16
    "r_ear",        # 17
    "l_ear",        # 18
    "l_big_toe",    # 19
    "l_small_toe",  # 20
    "l_heel",       # 21
    "r_big_toe",    # 22
    "r_small_toe",  # 23
    "r_heel"        # 24
]


ANGLE_TRIPLETS = [
    ("r_elbow", "r_shoulder", "r_elbow", "r_wrist"),
    ("l_elbow", "l_shoulder", "l_elbow", "l_wrist"),
    ("r_knee", "r_hip", "r_knee", "r_ankle"),
    ("l_knee", "l_hip", "l_knee", "l_ankle"),
    ("r_shoulder", "neck", "r_shoulder", "r_elbow"),
    ("l_shoulder", "neck", "l_shoulder", "l_elbow"),
    ("r_hip", "mid_hip", "r_hip", "r_knee"),
    ("l_hip", "mid_hip", "l_hip", "l_knee"),
]


def load_p_matrices(path: Path) -> np.ndarray:
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if len(csv_files) != 4:
            raise ValueError(f"Expected 4 CSV files in {path}, got {len(csv_files)}")
        mats = []
        for f in csv_files:
            mat = np.loadtxt(f, delimiter=",")
            if mat.shape != (3, 4):
                raise ValueError(f"CSV {f} must be shape (3,4), got {mat.shape}")
            mats.append(mat)
        return np.stack(mats, axis=0)

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.shape != (4, 3, 4):
            raise ValueError(f"NPY must be shape (4,3,4), got {arr.shape}")
        return arr

    if path.suffix.lower() == ".csv":
        mat = np.loadtxt(path, delimiter=",")
        if mat.shape == (3, 4):
            raise ValueError("CSV contains a single 3x4 matrix; pass a directory with 4 CSVs or JSON/NPY with 4x3x4")
        raise ValueError(f"CSV must be provided as a directory of 4 files; got {mat.shape}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "P" in data:
        mats = data["P"]
    else:
        mats = data
    arr = np.array(mats, dtype=float)
    if arr.shape != (4, 3, 4):
        raise ValueError(f"JSON must be shape (4,3,4), got {arr.shape}")
    return arr


def load_pose_from_json(json_path: Path, person_index: int) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    people = data.get("people", [])
    if not people or person_index >= len(people):
        return np.full((25, 2), np.nan), np.full((25,), np.nan)
    pose = people[person_index].get("pose_keypoints_2d", [])
    pts = np.full((25, 2), np.nan)
    conf = np.full((25,), np.nan)
    for i in range(25):
        base = i * 3
        if base + 2 < len(pose):
            x, y, c = pose[base], pose[base + 1], pose[base + 2]
            pts[i, 0] = x
            pts[i, 1] = y
            conf[i] = c
    return pts, conf


def triangulate_point_dlt(points: List[Tuple[float, float]],
                           confs: List[float],
                           p_mats: np.ndarray,
                           conf_thresh: float) -> Tuple[np.ndarray, float]:
    rows = []
    used_confs = []
    for (x, y), c, P in zip(points, confs, p_mats):
        if c is None or math.isnan(c) or c < conf_thresh:
            continue
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            continue
        w = math.sqrt(max(c, 1e-6))
        rows.append(w * (x * P[2] - P[0]))
        rows.append(w * (y * P[2] - P[1]))
        used_confs.append(c)
    if len(rows) < 4:
        return np.array([np.nan, np.nan, np.nan]), float("nan")
    A = np.stack(rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-9:
        return np.array([np.nan, np.nan, np.nan]), float("nan")
    X = X[:3] / X[3]
    return X, float(np.mean(used_confs))


def smooth_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.full_like(arr, np.nan, dtype=float)
    T = arr.shape[0]
    for t in range(T):
        t0 = max(0, t - half)
        t1 = min(T, t + half + 1)
        chunk = arr[t0:t1]
        valid = ~np.isnan(chunk)
        if valid.any():
            out[t] = np.nanmean(chunk, axis=0)
    return out


def compute_angles(pts3d: np.ndarray) -> Tuple[List[str], np.ndarray]:
    name_to_idx = {n: i for i, n in enumerate(BODY_25_NAMES)}
    angles = []
    angle_names = []
    for angle_name, a, b, c in ANGLE_TRIPLETS:
        ia, ib, ic = name_to_idx[a], name_to_idx[b], name_to_idx[c]
        A = pts3d[:, ia]
        B = pts3d[:, ib]
        C = pts3d[:, ic]
        BA = A - B
        BC = C - B
        dot = np.einsum("ij,ij->i", BA, BC)
        norm = np.linalg.norm(BA, axis=1) * np.linalg.norm(BC, axis=1)
        ang = np.full((pts3d.shape[0],), np.nan)
        valid = norm > 1e-9
        cosv = np.clip(dot[valid] / norm[valid], -1.0, 1.0)
        ang[valid] = np.degrees(np.arccos(cosv))
        angles.append(ang)
        angle_names.append(angle_name)
    return angle_names, np.stack(angles, axis=1)


def finite_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if arr.shape[0] < 2:
        return out
    out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
    out[0] = (arr[1] - arr[0]) / dt
    out[-1] = (arr[-1] - arr[-2]) / dt
    return out


def write_keypoints_csv(path: Path, frame_names: List[str], pts3d: np.ndarray, confs: np.ndarray) -> None:
    header = ["frame_index", "frame_name"]
    for name in BODY_25_NAMES:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_conf"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, frame in enumerate(frame_names):
            row = [i, frame]
            for j in range(25):
                x, y, z = pts3d[i, j]
                c = confs[i, j]
                row.extend([x, y, z, c])
            writer.writerow(row)


def write_angles_csv(path: Path, frame_names: List[str], angle_names: List[str], angles: np.ndarray) -> None:
    header = ["frame_index", "frame_name"] + angle_names
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, frame in enumerate(frame_names):
            row = [i, frame] + list(angles[i])
            writer.writerow(row)


def write_kinematics_csv(path: Path, frame_names: List[str], speeds: np.ndarray, accels: np.ndarray) -> None:
    header = ["frame_index", "frame_name"]
    for name in BODY_25_NAMES:
        header.extend([f"{name}_speed", f"{name}_accel"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, frame in enumerate(frame_names):
            row = [i, frame]
            for j in range(25):
                row.extend([speeds[i, j], accels[i, j]])
            writer.writerow(row)


def write_summary_csv(path: Path, angle_names: List[str], angles: np.ndarray, speeds: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "min", "max"])
        for idx, name in enumerate(angle_names):
            vals = angles[:, idx]
            writer.writerow([
                f"angle_{name}",
                np.nanmean(vals),
                np.nanstd(vals),
                np.nanmin(vals),
                np.nanmax(vals),
            ])
        for j, name in enumerate(BODY_25_NAMES):
            vals = speeds[:, j]
            writer.writerow([
                f"speed_{name}",
                np.nanmean(vals),
                np.nanstd(vals),
                np.nanmin(vals),
                np.nanmax(vals),
            ])


def main() -> None:
    ap = argparse.ArgumentParser(description="DLT 3D reconstruction + kinematics from OpenPose JSONs.")
    ap.add_argument("--views", nargs=4, required=True, help="4 view directories of OpenPose JSONs")
    ap.add_argument("--p-mats", required=True, help="Path to P matrices (JSON/NPY 4x3x4 or directory of 4 CSVs)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    ap.add_argument("--person-index", type=int, default=0, help="Person index in OpenPose JSON")
    ap.add_argument("--conf-thresh", type=float, default=0.2, help="Min 2D confidence to use")
    ap.add_argument("--smooth", type=int, default=5, help="Moving average window (frames)")
    args = ap.parse_args()

    view_dirs = [Path(v) for v in args.views]
    for d in view_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"View dir not found: {d}")

    p_mats = load_p_matrices(Path(args.p_mats))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(view_dirs[0].glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {view_dirs[0]}")

    frame_names = []
    pts3d_list = []
    confs_list = []

    for jf in json_files:
        frame_name = jf.name
        view_paths = [d / frame_name for d in view_dirs]
        if not all(p.exists() for p in view_paths):
            continue
        pts_views = []
        conf_views = []
        for vp in view_paths:
            pts, conf = load_pose_from_json(vp, args.person_index)
            pts_views.append(pts)
            conf_views.append(conf)
        pts_views = np.stack(pts_views, axis=0)  # (4,25,2)
        conf_views = np.stack(conf_views, axis=0)  # (4,25)

        frame_xyz = np.full((25, 3), np.nan)
        frame_conf = np.full((25,), np.nan)
        for j in range(25):
            points = [tuple(pts_views[v, j]) for v in range(4)]
            confs = [float(conf_views[v, j]) for v in range(4)]
            xyz, c = triangulate_point_dlt(points, confs, p_mats, args.conf_thresh)
            frame_xyz[j] = xyz
            frame_conf[j] = c
        frame_names.append(frame_name)
        pts3d_list.append(frame_xyz)
        confs_list.append(frame_conf)

    if not pts3d_list:
        raise RuntimeError("No synchronized frames found across all 4 views.")

    pts3d = np.stack(pts3d_list, axis=0)
    confs = np.stack(confs_list, axis=0)

    # smoothing
    pts3d_sm = pts3d.copy()
    for j in range(25):
        pts3d_sm[:, j, 0] = smooth_moving_average(pts3d[:, j, 0], args.smooth)
        pts3d_sm[:, j, 1] = smooth_moving_average(pts3d[:, j, 1], args.smooth)
        pts3d_sm[:, j, 2] = smooth_moving_average(pts3d[:, j, 2], args.smooth)

    angle_names, angles = compute_angles(pts3d_sm)

    dt = 1.0 / args.fps
    vel = finite_diff(pts3d_sm, dt)
    acc = finite_diff(vel, dt)
    speeds = np.linalg.norm(vel, axis=2)
    accels = np.linalg.norm(acc, axis=2)

    write_keypoints_csv(out_dir / "keypoints_3d.csv", frame_names, pts3d_sm, confs)
    write_angles_csv(out_dir / "angles.csv", frame_names, angle_names, angles)
    write_kinematics_csv(out_dir / "kinematics.csv", frame_names, speeds, accels)
    write_summary_csv(out_dir / "summary.csv", angle_names, angles, speeds)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
