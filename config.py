"""
Central configuration module for the X-ray object detection and XAI project.

This module defines all filesystem paths and runtime configuration that should
be shared across scripts. The goal is to eliminate hardcoded absolute paths
inside the code and make the project portable and easy to configure for
different environments (local machine, server, CI, etc.).

The YOLOv8 model loading, prediction and hyperparameter logic MUST NOT be
modified here. We only centralize file locations and general settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT: Path = Path(__file__).parent.resolve()


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths used across the project."""

    project_root: Path = PROJECT_ROOT

    # Model weights
    finetuned_weights: Path = PROJECT_ROOT / "runs_finetune" / "sixray_to_prohibited_finish" / "weights" / "finetuned_best.pt"

    # Datasets
    sixray_data_yaml: Path = PROJECT_ROOT / "Sixray.v4-yolo_data_640.yolov8" / "data.yaml"
    prohibited_items_data_yaml: Path = PROJECT_ROOT / "X-ray baggage detection.v1-prohibited_items.yolov8" / "data.yaml"

    # Default inputs
    default_xai_input: Path = PROJECT_ROOT / "xai_input.jpg"

    # Evaluation outputs
    catastrophic_eval_dir: Path = PROJECT_ROOT / "test_results_catastrophic_interference"

    # XAI base output directory (scripts will create timestamped subfolders)
    xai_outputs_root: Path = PROJECT_ROOT / "outputs" / "xai"


@dataclass(frozen=True)
class RuntimeConfig:
    """Non-path runtime configuration shared by multiple scripts."""

    confidence_threshold: float = 0.25
    image_size: int = 640


def get_timestamped_xai_output_dir(base_dir: Optional[Path] = None) -> Path:
    """
    Return a timestamped directory path under the XAI outputs root.

    The directory is not created here; callers are responsible for creating it
    when appropriate. This avoids accidental filesystem writes during imports.
    """

    base = base_dir or PathsConfig().xai_outputs_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / timestamp


PATHS = PathsConfig()
RUNTIME = RuntimeConfig()


