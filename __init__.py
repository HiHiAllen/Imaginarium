"""
Imaginarium: Vision-guided High-Quality 3D Scene Layout Generation
==================================================================

A complete pipeline for generating 3D scene layouts from single images.

完整的单图3D场景布局生成流水线。

主要组件 / Main Components:
- GeometryModule: 深度估计与点云生成 / Depth estimation & point cloud generation
- SemanticParsingModule: 场景语义解析 / Scene semantic parsing
- RetrievalModule: 3D资产检索 / 3D asset retrieval
- PoseModule: 物体姿态估计 / Object pose estimation
- LayoutModule: 布局优化 / Layout optimization

用法 / Usage:
    >>> from imaginarium import ImaginariumPipeline
    >>> pipeline = ImaginariumPipeline("demo/demo_4.png", debug=False)
    >>> pipeline.run()

或使用命令行 / Or use CLI:
    $ python run_imaginarium.py demo/demo_4.png --debug
"""

__version__ = "1.0.0"
__author__ = "Imaginarium Team"

# Core exports
from .core import Context, Config
from .pipeline import ImaginariumPipeline
from .modules import (
    GeometryModule,
    SemanticParsingModule,
    RetrievalModule,
    PoseModule,
    LayoutModule,
)

__all__ = [
    # Core
    'Context',
    'Config',
    # Pipeline
    'ImaginariumPipeline',
    # Modules
    'GeometryModule',
    'SemanticParsingModule',
    'RetrievalModule',
    'PoseModule',
    'LayoutModule',
]
