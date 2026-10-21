"""Visualize HLoc retrieval rankings and local feature matches."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import h5py
import numpy as np


def h5_image_names(path: Path) -> list[str]:
    names = set()
    with h5py.File(path, "r") as handle:
        def visit(name, item):
            if isinstance(item, h5py.Dataset) and item.parent.name != "/":
                names.add(item.parent.name.lstrip("/"))
        handle.visititems(visit)
    return sorted(names)


def discover(folder: Path, pattern: str, label: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {label} found in {folder} (pattern: {pattern})")
    return matches[0]


def discover_features(folder: Path, label: str) -> Path:
    matches = sorted(path for path in folder.glob("feats*.h5") if "combined" not in path.stem.lower())
    if not matches:
        raise FileNotFoundError(f"No {label} found in {folder} (expected feats*.h5, excluding *combined*)")
    return matches[0]


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resolve_image(root: Path, name: str) -> Path:
    path = root / Path(name.replace("\\", "/"))
    if path.is_file():
        return path
    # Some pipelines store HLoc keys relative to the dataset root, others relative
    # to its images directory.
    candidate = root / "images" / Path(name.replace("\\", "/"))
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Image key {name!r} was not found under {root} or {root / 'images'}")


def load_descriptors(path: Path, names: list[str]) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        vectors = [np.asarray(handle[name]["global_descriptor"], dtype=np.float32).reshape(-1) for name in names]
    matrix = np.stack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def load_pairs(path: Path, query_names: list[str], map_names: set[str]) -> dict[str, list[str]]:
    result = {name: [] for name in query_names}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2 and fields[0] in result and fields[1] in map_names:
                result[fields[0]].append(fields[1])
    return result


def retrieval_rankings(args, query_names: list[str], map_names: list[str]) -> dict[str, list[tuple[str, float]]]:
    pairs_path = args.pairs or next(iter(sorted((args.query / "hloc_outputs").glob("pairs*netvlad*.txt"))), None)
    if pairs_path and pairs_path.is_file():
        pairs = load_pairs(pairs_path, query_names, set(map_names))
        rankings = {}
        for query, candidates in pairs.items():
            # HLoc pair files preserve descending similarity order, but may not
            # include scores. Display rank as the source of ordering.
            rankings[query] = [(name, float(len(candidates) - i)) for i, name in enumerate(candidates)]
        if any(rankings.values()):
            return rankings

    query_desc_path = args.query_global or discover(args.query / "hloc_outputs", "global-feats*.h5", "query global descriptors")
    map_desc_path = args.map_global or discover(args.map / "hloc_outputs", "global-feats*.h5", "map global descriptors")
    qnames = [n for n in query_names if n in set(h5_image_names(query_desc_path))]
    mset = set(h5_image_names(map_desc_path))
    mnames = [n for n in map_names if n in mset]
    if not qnames or not mnames:
        raise ValueError("No common image keys found between image folders and global descriptor HDF5 files")
    qdesc, mdesc = load_descriptors(query_desc_path, qnames), load_descriptors(map_desc_path, mnames)
    scores = qdesc @ mdesc.T
    results = {}
    for row, query in enumerate(qnames):
        order = np.argsort(-scores[row])[: args.top_k]
        results[query] = [(mnames[i], float(scores[row, i])) for i in order]
    return results


def make_retrieval_mosaic(query_path: Path, candidates: list[tuple[str, float]], map_root: Path, output: Path, query_name: str):
    images = [read_rgb(query_path)] + [read_rgb(resolve_image(map_root, name)) for name, _ in candidates]
    target_h = 240
    tiles = []
    for index, (image, item) in enumerate(zip(images, [None] + candidates)):
        scale = target_h / image.shape[0]
        resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), target_h))
        title = f"QUERY: {query_name}" if index == 0 else f"TOP-{index}: {item[0]}  score={item[1]:.4f}"
        title_h = 34
        tile = np.full((target_h + title_h, resized.shape[1], 3), 250, dtype=np.uint8)
        tile[title_h:] = resized
        cv2.putText(tile, title[: max(10, resized.shape[1] // 9)], (7, 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (25, 25, 25), 1, cv2.LINE_AA)
        tiles.append(tile)
    width = max(tile.shape[1] for tile in tiles)
    canvas = np.full((len(tiles) * (target_h + 34), width, 3), 235, dtype=np.uint8)
    for row, tile in enumerate(tiles):
        canvas[row * tile.shape[0]:row * tile.shape[0] + tile.shape[0], :tile.shape[1]] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def read_pair_matches(path: Path, name0: str, name1: str):
    keys = [name0.replace("/", "-") + "/" + name1.replace("/", "-"),
            name1.replace("/", "-") + "/" + name0.replace("/", "-")]
    old = [name0.replace("/", "-") + "_" + name1.replace("/", "-"),
           name1.replace("/", "-") + "_" + name0.replace("/", "-")]
    with h5py.File(path, "r") as handle:
        key = next((key for key in keys + old if key in handle), None)
        if key is None:
            return None
        reverse = key in (keys[1], old[1])
        matches = np.asarray(handle[key]["matches0"], dtype=np.int64)
        idx0 = np.flatnonzero(matches >= 0)
        pairs = np.column_stack((idx0, matches[idx0]))
        return pairs[:, ::-1] if reverse else pairs


def feature_points(path: Path, name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[name]["keypoints"], dtype=np.float32)


def visualize_match(args, query_name: str, map_name: str, output: Path):
    match_file = args.matches or discover(args.query / "hloc_outputs", "matches*.h5", "query match file")
    query_features = args.query_features or discover_features(args.query / "hloc_outputs", "query local features")
    map_features = args.map_features or discover_features(args.map / "hloc_outputs", "map local features")
    matches = read_pair_matches(match_file, query_name, map_name)
    if matches is None:
        return False
    qpts, mpts = feature_points(query_features, query_name), feature_points(map_features, map_name)
    valid = (matches[:, 0] < len(qpts)) & (matches[:, 1] < len(mpts))
    matches = matches[valid]
    qxy, mxy = qpts[matches[:, 0]], mpts[matches[:, 1]]
    inliers = np.zeros(len(matches), dtype=bool)
    if len(matches) >= 8:
        _, mask = cv2.findFundamentalMat(qxy, mxy, cv2.FM_RANSAC, args.ransac_threshold, 0.999)
        if mask is not None:
            inliers = mask.ravel().astype(bool)
    left, right = read_rgb(resolve_image(args.query, query_name)), read_rgb(resolve_image(args.map, map_name))
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + right.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1]:] = right
    for i, (qp, mp) in enumerate(zip(qxy, mxy)):
        color = (40, 210, 70) if inliers[i] else (230, 65, 55)
        p0, p1 = tuple(np.round(qp).astype(int)), tuple(np.round(mp + [left.shape[1], 0]).astype(int))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)
    label = f"{query_name}  <->  {map_name} | matches={len(matches)} | inliers={int(inliers.sum())}"
    cv2.rectangle(canvas, (0, 0), (min(width, 1100), 32), (20, 20, 20), -1)
    cv2.putText(canvas, label[:150], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, default=Path(r"D:\Datasets\Indoor_query_38_38"), help="Query dataset root (contains images/ and hloc_outputs/)")
    parser.add_argument("--map", type=Path, default=Path(r"D:\Datasets\Indoor_map_51"), help="Map dataset root (contains images/ and hloc_outputs/)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("visualizations"))
    parser.add_argument("--pairs", type=Path, help="Optional HLoc retrieval pairs file")
    parser.add_argument("--query-global", type=Path)
    parser.add_argument("--map-global", type=Path)
    parser.add_argument("--query-features", type=Path)
    parser.add_argument("--map-features", type=Path)
    parser.add_argument("--matches", type=Path)
    parser.add_argument("--max-queries", type=int, default=0, help="Limit queries; 0 means all")
    parser.add_argument("--max-local-pairs", type=int, default=0, help="Render local matches for top-ranked pairs; 0 means all selected pairs")
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    qh5 = args.query_global or discover(args.query / "hloc_outputs", "global-feats*.h5", "query global descriptors")
    mh5 = args.map_global or discover(args.map / "hloc_outputs", "global-feats*.h5", "map global descriptors")
    query_names = h5_image_names(qh5)
    map_names = h5_image_names(mh5)
    rankings = retrieval_rankings(args, query_names, map_names)
    selected = list(rankings.items())
    if args.max_queries:
        selected = selected[:args.max_queries]
    args.output.mkdir(parents=True, exist_ok=True)
    local_count = 0
    for query_name, ranked in selected:
        ranked = ranked[:args.top_k]
        if not ranked:
            continue
        make_retrieval_mosaic(resolve_image(args.query, query_name), ranked, args.map, args.output / "retrieval" / (Path(query_name).stem + ".jpg"), query_name)
        for map_name, _ in ranked:
            if args.max_local_pairs and local_count >= args.max_local_pairs:
                break
            out = args.output / "matches" / (re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(query_name).stem + "__" + Path(map_name).stem) + ".jpg")
            if visualize_match(args, query_name, map_name, out):
                local_count += 1
    print(f"Saved retrieval visualizations for {len(selected)} queries and {local_count} local-match visualizations to {args.output}")


if __name__ == "__main__":
    main()
