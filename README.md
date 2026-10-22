# HLoc Visualization

用于查看 HLoc 的全局图像检索排序和局部特征匹配结果。脚本读取数据集 `hloc_outputs/` 中的 HDF5 特征、matches 文件及 pairs 文本，不重新运行特征提取或匹配。

## 目录结构

```text
visualize_Hloc/
├── visualize_hloc.py        # CLI 启动入口
├── hloc_visualizer/
│   ├── cli.py               # 参数解析、文件选择和任务编排
│   ├── io_utils.py          # HDF5 与图像读取、路径发现
│   ├── retrieval.py         # 全局描述子排序和检索拼图
│   └── matches.py           # 局部匹配、RANSAC 内点绘制
└── README.md
```

## 环境与运行

依赖使用 Conda `dl` 环境中已有的 `numpy`、`h5py` 和 `opencv-python`，不需要安装新包。默认数据路径为 `D:\Datasets\Indoor_query_38_38` 和 `D:\Datasets\Indoor_map_51`。

```powershell
conda run -n dl python visualize_hloc.py --retrieval netvlad --top-k 10 --output output/visualizations
```

选择 MegaLoc：

```powershell
conda run -n dl python visualize_hloc.py --retrieval megaloc --top-k 10 --output output/visualizations_megaloc
```

指定 LightGlue 匹配结果（自动选择时优先挑选与检索方法标签一致的 matches 文件）：

```powershell
conda run -n dl python visualize_hloc.py --retrieval megaloc --matcher lightglue --max-queries 5 --max-local-pairs 20
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--query`, `--map` | Query 与 map 数据集根目录，目录下应有 `images/` 和 `hloc_outputs/` |
| `--retrieval` | 全局检索类型：`netvlad` 或 `megaloc` |
| `--matcher` | matches 文件名中的匹配器标签，例如 `lightglue`；默认 `auto` |
| `--top-k` | 每张 query 展示的 map 候选数量 |
| `--output` | 输出目录 |
| `--max-queries` | 限制处理的 query 数量；`0` 表示全部 |
| `--max-local-pairs` | 限制局部匹配图数量；`0` 表示全部可用候选 |
| `--pairs` | 手动指定 HLoc retrieval pairs 文本文件 |
| `--query-global`, `--map-global` | 手动指定两侧全局特征 HDF5 |
| `--matches` | 手动指定局部匹配 HDF5 |
| `--query-features`, `--map-features` | 手动指定两侧局部特征 HDF5 |
| `--ransac-threshold` | 基础矩阵 RANSAC 的像素阈值 |

输出中 `retrieval/` 保存 query 与 Top-k map 图像的排名拼图；`matches/` 保存 query-map 匹配连线图。局部图显示总匹配数及基础矩阵 RANSAC 内点数，绿色为内点、红色为外点。若 HLoc 的 matches 文件没有对应图片对，该对只出现在检索拼图中，不会生成局部匹配图。

已有 `pairs-query-<method>*.txt` 时按文件中的顺序展示；该文件不包含相似度分数。没有 pairs 文件时，脚本从所选方法的全局描述子计算余弦相似度。
