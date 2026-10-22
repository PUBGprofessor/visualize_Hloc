"""Global-descriptor retrieval ranking and contact-sheet rendering."""

from pathlib import Path

import cv2
import numpy as np

from .io_utils import h5_image_names, load_descriptors, read_rgb, resolve_image


def read_pairs(path: Path, query_names: list[str], map_names: set[str]) -> dict[str, list[str]]:
    """Read HLoc's ordered 'query reference' pair list."""
    pairs = {name: [] for name in query_names}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2 and fields[0] in pairs and fields[1] in map_names:
                pairs[fields[0]].append(fields[1])
    return pairs


def rank_from_descriptors(query_file: Path, map_file: Path, top_k: int) -> dict[str, list[tuple[str, float]]]:
    """Compute cosine-similarity rankings when no pairs text file is supplied."""
    query_names, map_names = h5_image_names(query_file), h5_image_names(map_file)
    if not query_names or not map_names:
        raise ValueError("Global descriptor HDF5 file contains no image keys")
    query_desc = load_descriptors(query_file, query_names)
    map_desc = load_descriptors(map_file, map_names)
    scores = query_desc @ map_desc.T
    rankings = {}
    for row, query in enumerate(query_names):
        indices = np.argsort(-scores[row])[:top_k]
        rankings[query] = [(map_names[index], float(scores[row, index])) for index in indices]
    return rankings


def render_contact_sheet(query_path: Path, ranked: list[tuple[str, float | None]], map_root: Path, output: Path, query_name: str) -> None:
    """Save query and ranked map images as a labeled vertical contact sheet."""
    images = [read_rgb(query_path)] + [read_rgb(resolve_image(map_root, name)) for name, _ in ranked]
    tiles = []
    title_height, target_height = 34, 240
    for index, image in enumerate(images):
        scale = target_height / image.shape[0]
        resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), target_height))
        if index == 0:
            label = f"QUERY: {query_name}"
        else:
            map_name, score = ranked[index - 1]
            score_label = f"  score={score:.4f}" if score is not None else ""
            label = f"TOP-{index}: {map_name}{score_label}"
        tile = np.full((target_height + title_height, resized.shape[1], 3), 250, dtype=np.uint8)
        tile[title_height:] = resized
        cv2.putText(tile, label[: max(10, resized.shape[1] // 9)], (7, 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (25, 25, 25), 1, cv2.LINE_AA)
        tiles.append(tile)
    width = max(tile.shape[1] for tile in tiles)
    sheet = np.full((len(tiles) * tiles[0].shape[0], width, 3), 235, dtype=np.uint8)
    for row, tile in enumerate(tiles):
        sheet[row * tile.shape[0]:row * tile.shape[0] + tile.shape[0], :tile.shape[1]] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
