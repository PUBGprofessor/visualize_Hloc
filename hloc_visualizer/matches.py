"""Local-feature match visualization for HLoc matches HDF5 files."""

from pathlib import Path

import cv2
import h5py
import numpy as np

from .io_utils import read_rgb, resolve_image


def read_pair_matches(path: Path, name0: str, name1: str) -> np.ndarray | None:
    """Read HLoc matches0 in either image order and return index pairs."""
    key0 = name0.replace("/", "-")
    key1 = name1.replace("/", "-")
    keys = [f"{key0}/{key1}", f"{key1}/{key0}", f"{key0}_{key1}", f"{key1}_{key0}"]
    with h5py.File(path, "r") as handle:
        key = next((candidate for candidate in keys if candidate in handle), None)
        if key is None:
            return None
        reverse = key in (keys[1], keys[3])
        matches0 = np.asarray(handle[key]["matches0"], dtype=np.int64)
    query_indices = np.flatnonzero(matches0 >= 0)
    indices = np.column_stack((query_indices, matches0[query_indices]))
    return indices[:, ::-1] if reverse else indices


def load_keypoints(path: Path, name: str) -> np.ndarray:
    """Load Nx2 keypoint coordinates for one image key."""
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[name]["keypoints"], dtype=np.float32)


def render_matches(query_root: Path, map_root: Path, query_name: str, map_name: str,
                   matches_file: Path, query_features: Path, map_features: Path,
                   output: Path, ransac_threshold: float) -> bool:
    """Draw correspondences; green denotes a RANSAC geometric inlier."""
    indices = read_pair_matches(matches_file, query_name, map_name)
    if indices is None:
        return False
    query_points = load_keypoints(query_features, query_name)
    map_points = load_keypoints(map_features, map_name)
    valid = (indices[:, 0] < len(query_points)) & (indices[:, 1] < len(map_points))
    indices = indices[valid]
    qxy, mxy = query_points[indices[:, 0]], map_points[indices[:, 1]]
    inliers = np.zeros(len(indices), dtype=bool)
    if len(indices) >= 8:
        _fundamental, mask = cv2.findFundamentalMat(qxy, mxy, cv2.FM_RANSAC, ransac_threshold, 0.999)
        if mask is not None:
            inliers = mask.ravel().astype(bool)

    left = read_rgb(resolve_image(query_root, query_name))
    right = read_rgb(resolve_image(map_root, map_name))
    canvas = np.zeros((max(left.shape[0], right.shape[0]), left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1]:] = right
    for index, (query_point, map_point) in enumerate(zip(qxy, mxy)):
        color = (40, 210, 70) if inliers[index] else (230, 65, 55)
        start = tuple(np.round(query_point).astype(int))
        end = tuple(np.round(map_point + [left.shape[1], 0]).astype(int))
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, start, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, end, 3, color, -1, cv2.LINE_AA)
    label = f"{query_name} <-> {map_name} | matches={len(indices)} | inliers={int(inliers.sum())}"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 1100), 32), (20, 20, 20), -1)
    cv2.putText(canvas, label[:150], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return True
