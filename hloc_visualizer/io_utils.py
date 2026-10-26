"""HLoc 可视化共用的文件发现、HDF5 和图片读取工具。"""

from pathlib import Path

import cv2
import h5py
import numpy as np


def discover(folder: Path, pattern: str, description: str) -> Path:
    """在输出目录中按通配符寻找一个文件，并在缺失时给出可读错误。"""
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {description} found in {folder} (pattern: {pattern})")
    return matches[0]


def discover_local_features(folder: Path, explicit: Path | None, description: str) -> Path:
    """选择局部特征文件。

    查询侧可能同时存在普通特征文件和 ``*-combined.h5`` 辅助文件；后者
    不是一张图片一个组的标准局部特征输入，因此默认排除，用户也可通过
    ``--query-features``/``--map-features`` 显式指定文件。
    """
    if explicit:
        return explicit
    matches = sorted(path for path in folder.glob("feats*.h5") if "combined" not in path.stem.lower())
    if not matches:
        raise FileNotFoundError(f"No {description} found in {folder}")
    return matches[0]


def h5_image_names(path: Path) -> list[str]:
    """返回 HLoc HDF5 中的图片键名。

    HLoc 的特征组通常形如 ``/frame00001.png/keypoints``。遍历所有数据集，
    取其父组名称，可以兼容图片键中包含目录的情况；集合去重后排序，保证
    后续描述子矩阵的行顺序稳定。
    """
    names: set[str] = set()
    with h5py.File(path, "r") as handle:
        def collect(_name, item):
            if isinstance(item, h5py.Dataset) and item.parent.name != "/":
                names.add(item.parent.name.lstrip("/"))
        handle.visititems(collect)
    return sorted(names)


def resolve_image(root: Path, name: str) -> Path:
    """把 HLoc 图片键解析为实际文件路径。

    pairs.txt 常写 ``frame00001.png``，也可能写 ``images/frame00001.png``；
    因此同时尝试数据集根目录和 ``images/`` 子目录。
    """
    relative = Path(name.replace("\\", "/"))
    for candidate in (root / relative, root / "images" / relative):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image {name!r} was not found under {root}")


def read_rgb(path: Path) -> np.ndarray:
    """以 RGB 读取图片，并关闭 OpenCV 的隐式 EXIF 方向变换。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_descriptors(path: Path, names: list[str]) -> np.ndarray:
    """读取全局描述子并做 L2 归一化。

    归一化后，两个向量的点积就是余弦相似度；这与 HLoc 全局检索使用的
    常见排序方式一致。HDF5 中的 float16 在这里转成 float32，避免计算时
    发生不必要的精度损失。
    """
    with h5py.File(path, "r") as handle:
        rows = [np.asarray(handle[name]["global_descriptor"], dtype=np.float32).reshape(-1) for name in names]
    descriptors = np.stack(rows)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return descriptors / np.maximum(norms, 1e-12)
