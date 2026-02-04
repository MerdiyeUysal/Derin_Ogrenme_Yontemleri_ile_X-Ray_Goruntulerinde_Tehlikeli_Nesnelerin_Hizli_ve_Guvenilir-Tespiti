"""Command-line runner for Grad-CAM based XAI on top of a YOLOv8 model.

This script orchestrates the end-to-end workflow:

- load a pre-trained YOLOv8 model (weights and inference parameters are kept
  exactly as originally used in the project),
- run object detection on a single input X-ray image,
- compute Grad-CAM explanations for the most confident detection, and
- write multiple visualization artifacts and metadata to disk.

The core YOLOv8 model loading, prediction behaviour and hyperparameters are
intentionally preserved; this file focuses on robustness, readability and
project-wide configuration.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import torch
import torch.nn as nn

# Modül import'u için path düzeltmesi
# Script olarak çalıştırıldığında parent dizini path'e ekle
script_dir = Path(__file__).parent.parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from ultralytics import YOLO

from config import PATHS, RUNTIME, get_timestamped_xai_output_dir

# Import stratejisi: önce relative, sonra absolute
try:
    from .gradcam_yolov8 import GradCAM, find_last_conv, compute_differentiable_score
    from .utils import (
        ensure_output_dir,
        clean_output_dir,
        load_image,
        preprocess_image_for_yolo,
        overlay_heatmap,
        overlay_heatmap_full,
        overlay_heatmap_bbox,
        draw_detections,
        save_heatmap_only,
        save_metadata,
        format_detection_info
    )
except (ImportError, ValueError):
    # Relative import başarısız olursa absolute import dene
    try:
        from xai.gradcam_yolov8 import GradCAM, find_last_conv, compute_differentiable_score
        from xai.utils import (
            ensure_output_dir,
            clean_output_dir,
            load_image,
            preprocess_image_for_yolo,
            overlay_heatmap,
            overlay_heatmap_full,
            overlay_heatmap_bbox,
            draw_detections,
            save_heatmap_only,
            save_metadata,
            format_detection_info
        )
    except ImportError:
        # Son çare: doğrudan modül import
        import importlib.util
        spec1 = importlib.util.spec_from_file_location(
            "gradcam_yolov8", 
            Path(__file__).parent / "gradcam_yolov8.py"
        )
        gradcam_module = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(gradcam_module)
        
        spec2 = importlib.util.spec_from_file_location(
            "utils",
            Path(__file__).parent / "utils.py"
        )
        utils_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(utils_module)
        
        GradCAM = gradcam_module.GradCAM
        find_last_conv = gradcam_module.find_last_conv
        compute_differentiable_score = gradcam_module.compute_differentiable_score
        ensure_output_dir = utils_module.ensure_output_dir
        clean_output_dir = utils_module.clean_output_dir
        load_image = utils_module.load_image
        preprocess_image_for_yolo = utils_module.preprocess_image_for_yolo
        overlay_heatmap = utils_module.overlay_heatmap
        overlay_heatmap_full = utils_module.overlay_heatmap_full
        overlay_heatmap_bbox = utils_module.overlay_heatmap_bbox
        draw_detections = utils_module.draw_detections
        save_heatmap_only = utils_module.save_heatmap_only
        save_metadata = utils_module.save_metadata
        format_detection_info = utils_module.format_detection_info


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the XAI CLI.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including image path, weights path, confidence
        threshold, image size and output directory.
    """

    parser = argparse.ArgumentParser(description="YOLOv8 XAI: Grad-CAM based explainability")
    parser.add_argument(
        "--image",
        type=str,
        default=str(PATHS.default_xai_input),
        help="Input image path (default: project-level xai_input.jpg)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PATHS.finetuned_weights),
        help="YOLOv8 model weights path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=RUNTIME.confidence_threshold,
        help="Confidence threshold (default comes from central runtime config)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=RUNTIME.image_size,
        help="Square image size used for inference (default from runtime config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory. If omitted, a timestamped folder is created under outputs/xai/",
    )
    parser.add_argument(
        "--no-clean",
        dest="clean_output",
        action="store_false",
        default=True,
        help="Çıktı klasörünü temizleme (eski dosyaları korur). Varsayılan: temizleme aktif"
    )
    return parser.parse_args()


def run_xai(
    image_path: str,
    weights_path: str,
    conf_threshold: float = RUNTIME.confidence_threshold,
    imgsz: int = RUNTIME.image_size,
    output_dir: Optional[str] = None,
    clean_output: bool = True,
) -> None:
    """
    Execute the main XAI workflow for a single image.

    The function keeps the original YOLOv8 behaviour intact while adding
    safer path handling, device checks and logging-friendly messages.

    Parameters
    ----------
    image_path:
        Path to the input X-ray image.
    weights_path:
        Path to the YOLOv8 weights file.
    conf_threshold:
        Confidence threshold for YOLOv8 predictions.
    imgsz:
        Square image size used for inference.
    output_dir:
        Output directory where visualizations and metadata will be written.
        If ``None``, a timestamped directory under ``outputs/xai`` is used.
    clean_output:
        If ``True``, remove old files in the chosen output directory before
        writing new ones.
    """

    # === PATH NORMALIZATION ===
    image_path_obj = Path(image_path)
    if not image_path_obj.is_absolute():
        image_path_obj = PATHS.project_root / image_path_obj
    image_path_obj = image_path_obj.resolve()

    if not image_path_obj.exists():
        raise FileNotFoundError(f"Input image not found: {image_path_obj}")

    weights_path_obj = Path(weights_path)
    if not weights_path_obj.is_absolute():
        weights_path_obj = PATHS.project_root / weights_path_obj
    weights_path_obj = weights_path_obj.resolve()

    if not weights_path_obj.exists():
        raise FileNotFoundError(f"Model weights file not found: {weights_path_obj}")

    print(f"MODEL: {weights_path_obj.absolute()} EXISTS: {weights_path_obj.exists()}")

    # === OUTPUT DIRECTORY HANDLING ===
    if output_dir is None:
        out_dir = get_timestamped_xai_output_dir()
    else:
        out_dir = Path(output_dir)

    ensure_output_dir(str(out_dir))
    print(f"OUT_DIR: {out_dir.absolute()}")

    # === CLEAN PREVIOUS OUTPUTS (OPTIONAL) ===
    if clean_output:
        print("🧹 Cleaning existing outputs in directory...")
        clean_output_dir(str(out_dir))

    # === DEVICE SELECTION ===
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as error:
        print(f"⚠️  Failed to determine CUDA availability, falling back to CPU: {error}")
        device = torch.device("cpu")

    print(f"DEVICE: {device}")

    # === MODEL LOADING (CORE LOGIC UNCHANGED) ===
    print("Model is being loaded...")
    try:
        yolo = YOLO(str(weights_path_obj))
    except Exception as error:
        raise RuntimeError(f"Failed to load YOLOv8 model from {weights_path_obj}") from error

    yolo_model = yolo.model  # torch.nn.Module
    
    # Core model'e erişim
    # YOLO model yapısı: yolo.model -> DetectionModel -> model (nn.Sequential)
    # Bazı YOLO versiyonlarında yapı farklı olabilir, bu yüzden güvenli erişim
    if hasattr(yolo_model, 'model'):
        core_model = yolo_model.model
    else:
        # Alternatif: doğrudan yolo_model'i kullan
        core_model = yolo_model
    
    core_model.to(device)
    
    # Backbone'u belirle (Concat modüllerinden önceki kısmı)
    from ultralytics.nn.modules import Concat
    if isinstance(core_model, nn.Sequential):
        layers = list(core_model.children())
        backbone_layers = []
        for layer in layers:
            if isinstance(layer, Concat):
                break
            backbone_layers.append(layer)
        if not backbone_layers:
            backbone_layers = layers[:20] if len(layers) > 20 else layers
        backbone_model = nn.Sequential(*backbone_layers).to(device)
    elif hasattr(core_model, 'backbone'):
        backbone_model = core_model.backbone
    else:
        backbone_model = core_model
    
    # Grad-CAM için son Conv2d katmanını backbone içinde bul
    print("Son Conv2d katmanı aranıyor...")
    target_layer = find_last_conv(backbone_model)
    print(f"Target layer bulundu: {target_layer}")
    
    # Grad-CAM engine'i oluştur (backbone_model kullan)
    cam_engine = GradCAM(backbone_model, target_layer)
    
    # === GÖRÜNTÜ YÜKLEME ===
    print(f"Loading image: {image_path_obj}")
    import hashlib
    import shutil
    import time

    abs_img = os.path.abspath(str(image_path_obj))
    print("IMAGE_ABS_PATH:", abs_img)
    print("IMAGE_EXISTS:", os.path.exists(abs_img))
    print("IMAGE_MTIME:", time.ctime(os.path.getmtime(abs_img)) if os.path.exists(abs_img) else None)
    print("IMAGE_SIZE_BYTES:", os.path.getsize(abs_img) if os.path.exists(abs_img) else None)

    # hash hesapla
    if os.path.exists(abs_img):
        with open(abs_img, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        print("IMAGE_SHA256:", sha)

    # çıktı klasörüne input kopyası
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(abs_img):
        shutil.copy2(abs_img, os.path.join(out_dir, "input_copy.jpg"))
        print("✅ input_copy.jpg yazıldı (kanıt):", os.path.join(out_dir, "input_copy.jpg"))

    bgr_img = load_image(str(image_path_obj))
    h_orig, w_orig = bgr_img.shape[:2]
    print(f"Görüntü boyutu: {w_orig}x{h_orig}")
    
    # === YOLO TAHMİN (Bbox'ları almak için) ===
    print(f"YOLO tahmin çalıştırılıyor (conf={conf_threshold})...")
    with torch.no_grad():
    
        results = yolo(
            str(image_path),
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False
    )

    
    # Tespit kontrolü
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        print(f"⚠️  Hiç tespit yok (conf={conf_threshold}).")
        print(f"💡 Daha düşük conf değeri deneyin (örn. --conf 0.10) veya farklı görüntü seçin.")
        return        
    
    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    best_idx = int(confs.argmax())
    best_box = boxes[best_idx]
    
    # En yüksek confidence'lı bbox bilgileri
    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
    best_conf = float(best_box.conf.cpu().item())
    best_cls = int(best_box.cls.cpu().item())
    cls_name = results[0].names.get(best_cls, str(best_cls))
    
    print(f"✅ En güçlü tespit: class={cls_name} conf={best_conf:.3f} bbox=({x1},{y1},{x2},{y2})")
    
    # Tüm tespitleri metadata için topla
    all_detections = []
    all_boxes = []
    all_confs = []
    all_class_names = []
    
    for i in range(len(boxes)):
        box = boxes[i]
        x1_i, y1_i, x2_i, y2_i = box.xyxy[0].cpu().numpy().astype(int)
        conf_i = float(box.conf.cpu().item())
        cls_i = int(box.cls.cpu().item())
        cls_name_i = results[0].names.get(cls_i, str(cls_i))
        
        all_detections.append(format_detection_info(
            x1_i, y1_i, x2_i, y2_i, conf_i, cls_i, cls_name_i
        ))
        all_boxes.append((x1_i, y1_i, x2_i, y2_i))
        all_confs.append(conf_i)
        all_class_names.append(cls_name_i)
    
    # === GRAD-CAM HESAPLAMA ===
    print("Grad-CAM hesaplanıyor...")
    
    # Görüntüyü preprocess et
    rgb_img, img_tensor = preprocess_image_for_yolo(bgr_img, target_size=imgsz)
    img_tensor = img_tensor.to(device)
    
    # Differentiable skor hesapla (backbone_model kullan)
    # Tespit edilen nesnenin bbox'ına göre score hesapla
    print("Differentiable skor hesaplanıyor...")
    target_bbox = (x1, y1, x2, y2)
    score = compute_differentiable_score(backbone_model, img_tensor, device, target_bbox)
    print(f"SCORE requires_grad: {score.requires_grad}")
    print(f"SCORE grad_fn: {score.grad_fn}")
    
    if not score.requires_grad:
        raise RuntimeError("Skor requires_grad=False! Grad-CAM çalışmayacak.")
    
    # Grad-CAM üret
    cam = cam_engine(score)
    print(f"Grad-CAM üretildi, boyut: {cam.shape}")
    
    # === ÇIKTILARI KAYDET ===
    
    # 1. detections.jpg (bbox + label)
    detections_img = draw_detections(bgr_img, all_boxes, all_confs, all_class_names)
    detections_path = out_dir / "detections.jpg"
    cv2.imwrite(str(detections_path), detections_img)
    print(f"✅ Kaydedildi: {detections_path}")
    
    # 2. heatmap.jpg (sadece heatmap)
    heatmap_path = out_dir / "heatmap.jpg"
    save_heatmap_only(cam, str(heatmap_path), (h_orig, w_orig))
    print(f"✅ Kaydedildi: {heatmap_path}")
    
    # 3. xai_heatmap_overlay.jpg (heatmap + bbox overlay) - GERİYE DÖNÜK UYUMLULUK
    overlay = overlay_heatmap(bgr_img, cam, alpha=0.5)
    # En yüksek confidence'lı bbox'ı vurgula (daha kalın ve belirgin)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
    # Label arka planı
    label = f"{cls_name} {best_conf:.2f}"
    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    cv2.rectangle(
        overlay,
        (max(0, x1), max(0, y1 - label_h - 15)),
        (x1 + label_w + 10, y1 + 5),
        (0, 255, 0),
        -1
    )
    cv2.putText(
        overlay,
        label,
        (max(5, x1 + 5), max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),  # Siyah text daha okunabilir
        2,
        cv2.LINE_AA
    )
    overlay_path = out_dir / "xai_heatmap_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"✅ Kaydedildi: {overlay_path}")
    
    # 4. xai_overlay_full.jpg (İYİLEŞTİRİLMİŞ: Tam görüntü + kontrollü heatmap)
    overlay_full = overlay_heatmap_full(bgr_img, cam, boxes=all_boxes, alpha_base=0.35, alpha_bbox=0.7, bbox_focus_strength=0.25)
    # Bbox'ları çiz
    for (x1_i, y1_i, x2_i, y2_i), conf_i, cls_name_i in zip(all_boxes, all_confs, all_class_names):
        # En yüksek confidence'lı bbox'ı daha kalın çiz
        thickness = 4 if (x1_i, y1_i, x2_i, y2_i) == (x1, y1, x2, y2) else 2
        color = (0, 255, 0) if (x1_i, y1_i, x2_i, y2_i) == (x1, y1, x2, y2) else (0, 255, 255)
        cv2.rectangle(overlay_full, (x1_i, y1_i), (x2_i, y2_i), color, thickness)
        # Label
        label_i = f"{cls_name_i} {conf_i:.2f}"
        (label_w_i, label_h_i), _ = cv2.getTextSize(label_i, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            overlay_full,
            (max(0, x1_i), max(0, y1_i - label_h_i - 10)),
            (x1_i + label_w_i + 8, y1_i + 3),
            color,
            -1
        )
        cv2.putText(
            overlay_full,
            label_i,
            (max(3, x1_i + 3), max(18, y1_i - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
    overlay_full_path = out_dir / "xai_overlay_full.jpg"
    cv2.imwrite(str(overlay_full_path), overlay_full)
    print(f"✅ Kaydedildi: {overlay_full_path}")
    
    # 5. xai_overlay_bbox.jpg (İYİLEŞTİRİLMİŞ: Sadece bbox crop + yüksek kontrastlı heatmap)
    overlay_bbox = overlay_heatmap_bbox(bgr_img, cam, (x1, y1, x2, y2), padding=25, alpha=0.75)
    # En yüksek confidence'lı bbox'ı vurgula
    cv2.rectangle(overlay_bbox, (x1, y1), (x2, y2), (0, 255, 0), 4)
    # Label
    label_bbox = f"{cls_name} {best_conf:.2f}"
    (label_w_bbox, label_h_bbox), _ = cv2.getTextSize(label_bbox, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(
        overlay_bbox,
        (max(0, x1), max(0, y1 - label_h_bbox - 15)),
        (x1 + label_w_bbox + 10, y1 + 5),
        (0, 255, 0),
        -1
    )
    cv2.putText(
        overlay_bbox,
        label_bbox,
        (max(5, x1 + 5), max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    overlay_bbox_path = out_dir / "xai_overlay_bbox.jpg"
    cv2.imwrite(str(overlay_bbox_path), overlay_bbox)
    print(f"✅ Kaydedildi: {overlay_bbox_path}")
    
    # 6. meta.json (opsiyonel)
    meta_path = out_dir / "meta.json"
    save_metadata(str(meta_path), all_detections, conf_threshold)
    print(f"✅ Kaydedildi: {meta_path}")
    
    print("\n🎉 XAI analizi tamamlandı!")


def main():
    """CLI entry point."""
    args = parse_args()
    try:
        run_xai(
            image_path=args.image,
            weights_path=args.weights,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            output_dir=args.output_dir,
            clean_output=args.clean_output
        )
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())

