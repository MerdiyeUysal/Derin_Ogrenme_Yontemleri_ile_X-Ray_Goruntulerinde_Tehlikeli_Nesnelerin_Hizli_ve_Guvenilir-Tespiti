"""
XAI Modülü: YOLOv8 için Grad-CAM tabanlı açıklanabilirlik.
"""
__version__ = "1.0.0"

from .gradcam_yolov8 import GradCAM, find_last_conv, compute_differentiable_score
from .utils import (
    overlay_heatmap,
    draw_detections,
    save_heatmap_only,
    load_image,
    preprocess_image_for_yolo
)

__all__ = [
    "GradCAM",
    "find_last_conv",
    "compute_differentiable_score",
    "overlay_heatmap",
    "draw_detections",
    "save_heatmap_only",
    "load_image",
    "preprocess_image_for_yolo",
]

