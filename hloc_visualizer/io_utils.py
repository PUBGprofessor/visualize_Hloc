"""Shared HDF5 and image I/O helpers for HLoc visualizations."""

from pathlib import Path

import cv2
import h5py
import numpy as np


def discover(folder: Path, pattern: str, description: str) -> Path:
    """Find one matching output file, failing with a useful message if absent."""
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {description} found in {folder} (pattern: {pattern})")
    return matches[0]


def discover_local_features(folder: Path, explicit: Path | None, description: str) -> Path:
    """Ignore auxiliary '*combined*' files when selecting local features."""
    if explicit:
        return explicit
    matches = sorted(path for path in folder.glob("feats*.h5") if "combined" not in path.stem.lower())
    if not matches:
        raise FileNotFoundError(f"No {description} found in {folder}")
    return matches[0]


def h5_image_names(path: Path) -> list[str]:
    """Return image keys from a HLoc HDF5 file."""
    names: set[str] = set()
    with h5py.File(path, "r") as handle:
        def collect(_name, item):
            if isinstance(item, h5py.Dataset) and item.parent.name != "/":
                names.add(item.parent.name.lstrip("/"))
        handle.visititems(collect)
    return sorted(names)


def resolve_image(root: Path, name: str) -> Path:
    """Resolve either dataset-relative or images-directory-relative HLoc keys."""
    relative = Path(name.replace("\\", "/"))
    for candidate in (root / relative, root / "images" / relative):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image {name!r} was not found under {root}")


def read_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB without applying EXIF orientation implicitly."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_descriptors(path: Path, names: list[str]) -> np.ndarray:
    """Load and L2-normalize global descriptors for cosine ranking."""
    with h5py.File(path, "r") as handle:
        rows = [np.asarray(handle[name]["global_descriptor"], dtype=np.float32).reshape(-1) for name in names]
    descriptors = np.stack(rows)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return descriptors / np.maximum(norms, 1e-12)
