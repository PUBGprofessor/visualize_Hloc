"""HLoc 可视化脚本的最小启动入口。

实际参数解析和任务编排位于 ``hloc_visualizer.cli``；保留这一层是为了让
用户可以直接运行 ``python visualize_hloc.py ...``。
"""

from hloc_visualizer.cli import main


if __name__ == "__main__":
    main()
