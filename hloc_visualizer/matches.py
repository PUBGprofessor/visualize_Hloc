"""局部特征匹配的读取、几何过滤和可视化。

HLoc 的匹配文件通常使用如下 HDF5 层级：
``/query_image/map_image/matches0``。其中 ``matches0[i]`` 是
query 图第 ``i`` 个关键点在 map 图中的匹配下标，``-1`` 表示没有匹配。
"""

from pathlib import Path

import cv2
import h5py
import numpy as np

from .io_utils import read_rgb, resolve_image


def read_pair_matches(path: Path, name0: str, name1: str) -> np.ndarray | None:
    """读取一对图片的匹配索引，并统一返回 ``[query_index, map_index]``。

    不同版本/脚本可能把图片对保存为嵌套组、带 ``-`` 的扁平键或带 ``_``
    的扁平键。优先按真实 HLoc 输出的嵌套组读取，同时兼容后两种形式。
    如果文件只保存了反向顺序，则把列交换回来，保证调用方无需关心文件
    内部的图片顺序。
    """
    # HLoc 当前常见格式是 /image0/image1/matches0；图片名中的目录分隔符
    # 在 HDF5 中天然就是组层级，因此不能只用 ``"image0-image1"`` 查根键。
    flat0 = name0.replace("/", "-")
    flat1 = name1.replace("/", "-")
    with h5py.File(path, "r") as handle:
        candidates = [
            (name0, name1, False),
            (name1, name0, True),
        ]
        for first, second, reverse in candidates:
            # 嵌套组格式：handle[first][second]["matches0"]。
            if first in handle and isinstance(handle[first], h5py.Group) and second in handle[first]:
                group = handle[first][second]
                if "matches0" in group:
                    matches0 = np.asarray(group["matches0"], dtype=np.int64)
                    query_indices = np.flatnonzero(matches0 >= 0)
                    indices = np.column_stack((query_indices, matches0[query_indices]))
                    return indices[:, ::-1] if reverse else indices

        # 兼容某些旧导出脚本的扁平路径/键名。
        flat_candidates = [
            (f"{flat0}/{flat1}", False), (f"{flat1}/{flat0}", True),
            (f"{flat0}_{flat1}", False), (f"{flat1}_{flat0}", True),
        ]
        for key, reverse in flat_candidates:
            if key not in handle:
                continue
            group = handle[key]
            if "matches0" not in group:
                continue
            matches0 = np.asarray(group["matches0"], dtype=np.int64)
            query_indices = np.flatnonzero(matches0 >= 0)
            indices = np.column_stack((query_indices, matches0[query_indices]))
            return indices[:, ::-1] if reverse else indices
    return None


def load_keypoints(path: Path, name: str) -> np.ndarray:
    """从局部特征 HDF5 读取一张图片的 ``N×2`` 关键点坐标。"""
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[name]["keypoints"], dtype=np.float32)


def render_matches(query_root: Path, map_root: Path, query_name: str, map_name: str,
                   matches_file: Path, query_features: Path, map_features: Path,
                   output: Path, ransac_threshold: float) -> bool:
    """绘制匹配连线，并用基础矩阵 RANSAC 区分几何内点和外点。

    这里的 RANSAC 只是可视化辅助分析，不会修改 HLoc 的原始匹配文件，也
    不等同于 HLoc 最终定位阶段的 PnP 内点统计。
    """
    indices = read_pair_matches(matches_file, query_name, map_name)
    if indices is None:
        return False
    query_points = load_keypoints(query_features, query_name)
    map_points = load_keypoints(map_features, map_name)
    # 防御性检查：特征文件和匹配文件若不是同一次运行生成，可能出现越界索引。
    valid = (indices[:, 0] >= 0) & (indices[:, 1] >= 0)
    valid &= (indices[:, 0] < len(query_points)) & (indices[:, 1] < len(map_points))
    indices = indices[valid]
    qxy, mxy = query_points[indices[:, 0]], map_points[indices[:, 1]]
    inliers = np.zeros(len(indices), dtype=bool)
    # 基础矩阵至少需要 8 对点；不足时保留全部匹配，但不标记内点。
    if len(indices) >= 8:
        _fundamental, mask = cv2.findFundamentalMat(qxy, mxy, cv2.FM_RANSAC, ransac_threshold, 0.999)
        if mask is not None:
            inliers = mask.ravel().astype(bool)

    left = read_rgb(resolve_image(query_root, query_name))
    right = read_rgb(resolve_image(map_root, map_name))
    # 把 query 放左边、map 放右边，右图的 x 坐标需要加上左图宽度。
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
