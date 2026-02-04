"""
XAI modülü için yardımcı fonksiyonlar: görsel I/O, overlay, path utilities.
"""
import os
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def ensure_output_dir(output_dir: str) -> Path:
    """Çıktı klasörünü oluşturur ve Path döndürür."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def clean_output_dir(output_dir: str) -> None:
    """
    Çıktı klasöründeki tüm dosyaları siler.
    
    Args:
        output_dir: Temizlenecek çıktı klasörü yolu
    """
    out_path = Path(output_dir)
    if not out_path.exists():
        return
    
    # Klasördeki tüm dosyaları sil
    for file_path in out_path.iterdir():
        if file_path.is_file():
            try:
                file_path.unlink()
                print(f"🗑️  Silindi: {file_path.name}")
            except Exception as e:
                print(f"⚠️  Silinemedi {file_path.name}: {e}")
    
    print(f"✅ Çıktı klasörü temizlendi: {out_path}")


def load_image(image_path: str) -> np.ndarray:
    """
    Görüntüyü BGR formatında yükler.
    
    Args:
        image_path: Görüntü dosya yolu
        
    Returns:
        BGR görüntü (numpy array)
        
    Raises:
        FileNotFoundError: Görüntü bulunamazsa
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
    return img


def preprocess_image_for_yolo(
    bgr_img: np.ndarray, 
    target_size: int = 640
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Görüntüyü YOLO için ön işleme yapar.
    
    Args:
        bgr_img: BGR formatında görüntü
        target_size: Hedef boyut (kare)
        
    Returns:
        (orijinal_boyutlu_rgb, tensor) tuple
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    
    # Resize
    img_resized = cv2.resize(rgb, (target_size, target_size))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.transpose(img_resized, (2, 0, 1))  # CHW
    
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0)  # [1, C, H, W]
    
    return rgb, img_tensor


def enhance_heatmap(
    cam: np.ndarray,
    percentile_low: float = 80.0,
    percentile_high: float = 99.5,
    top_percent: float = 25.0,
    gaussian_blur: int = 5
) -> np.ndarray:
    """
    Heatmap'i iyileştirir: percentile clipping, threshold, Gaussian blur.
    
    Args:
        cam: Normalize edilmiş CAM (0-1 arası, [H, W])
        percentile_low: Alt percentile clipping (varsayılan: 80)
        percentile_high: Üst percentile clipping (varsayılan: 99.5)
        top_percent: Sadece en yüksek %X aktivasyonu tut (varsayılan: 25)
        gaussian_blur: Gaussian blur kernel boyutu (varsayılan: 5, 0 ise blur yok)
        
    Returns:
        İyileştirilmiş CAM (0-1 arası)
    """
    # NaN ve inf değerlerini temizle
    cam_clean = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
    cam_clean = np.clip(cam_clean, 0, 1)
    
    # Percentile clipping (daha agresif normalizasyon)
    if cam_clean.max() > cam_clean.min():
        p_low = np.percentile(cam_clean, percentile_low)
        p_high = np.percentile(cam_clean, percentile_high)
        
        # Clipping uygula
        cam_clipped = np.clip(cam_clean, p_low, p_high)
        # Min-max normalize et
        if p_high > p_low:
            cam_clipped = (cam_clipped - p_low) / (p_high - p_low + 1e-8)
        else:
            cam_clipped = cam_clean
    else:
        cam_clipped = cam_clean
    
    # Top %X threshold (düşük aktivasyonları bastır)
    if top_percent > 0 and cam_clipped.max() > 0:
        threshold = np.percentile(cam_clipped, 100 - top_percent)
        # Threshold altındaki değerleri zayıflat
        cam_clipped = np.where(
            cam_clipped >= threshold,
            cam_clipped,
            cam_clipped * 0.1  # Çok düşük aktivasyonları neredeyse sıfırla
        )
        # Yeniden normalize et
        if cam_clipped.max() > 0:
            cam_clipped = cam_clipped / cam_clipped.max()
    
    # Gaussian blur ile gürültüyü azalt
    if gaussian_blur > 0 and gaussian_blur % 2 == 1:
        cam_clipped = cv2.GaussianBlur(
            cam_clipped,
            (gaussian_blur, gaussian_blur),
            0
        )
        # Blur sonrası normalize et
        if cam_clipped.max() > 0:
            cam_clipped = cam_clipped / cam_clipped.max()
    
    return np.clip(cam_clipped, 0, 1)


def overlay_heatmap(
    bgr_img: np.ndarray, 
    cam: np.ndarray, 
    alpha: float = 0.6
) -> np.ndarray:
    """
    Heatmap'i görüntü üzerine bindirir (GERİYE DÖNÜK UYUMLULUK İÇİN KORUNDU).
    Yeni iyileştirilmiş versiyonlar için overlay_heatmap_full veya overlay_heatmap_bbox kullanın.
    
    Args:
        bgr_img: BGR formatında orijinal görüntü
        cam: Normalize edilmiş CAM (0-1 arası, [H, W])
        alpha: Overlay şeffaflığı (0-1)
        
    Returns:
        Overlay edilmiş BGR görüntü
    """
    h, w = bgr_img.shape[:2]
    
    # CAM'i görüntü boyutuna ölçekle (yüksek kaliteli interpolasyon)
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # İyileştirilmiş heatmap işleme
    cam_enhanced = enhance_heatmap(cam_resized, percentile_low=80.0, percentile_high=99.5, top_percent=25.0, gaussian_blur=5)
    
    # Normalize et (0-255 arası)
    cam_enhanced = (np.clip(cam_enhanced, 0, 1) * 255).astype(np.uint8)
    
    # TURBO colormap uygula (JET yerine, daha algısal olarak net)
    heatmap = cv2.applyColorMap(cam_enhanced, cv2.COLORMAP_TURBO)
    
    # Orijinal görüntüyü gri tonlara çevir (heatmap daha belirgin olsun)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Overlay: gri görüntü + renkli heatmap
    out = cv2.addWeighted(gray_bgr, 1 - alpha, heatmap, alpha, 0)
    
    return out


def overlay_heatmap_full(
    bgr_img: np.ndarray,
    cam: np.ndarray,
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    alpha_base: float = 0.4,
    alpha_bbox: float = 0.7,
    bbox_focus_strength: float = 0.3
) -> np.ndarray:
    """
    Tam görüntü üzerine kontrollü heatmap overlay (bbox odaklı).
    
    Args:
        bgr_img: BGR formatında orijinal görüntü
        cam: Normalize edilmiş CAM (0-1 arası, [H, W])
        boxes: Bbox listesi [(x1, y1, x2, y2), ...] (opsiyonel)
        alpha_base: Arka plan için alpha (varsayılan: 0.4)
        alpha_bbox: Bbox içi için alpha (varsayılan: 0.7)
        bbox_focus_strength: Bbox dışı heatmap zayıflatma gücü (0-1, varsayılan: 0.3)
        
    Returns:
        Overlay edilmiş BGR görüntü
    """
    h, w = bgr_img.shape[:2]
    
    # CAM'i görüntü boyutuna ölçekle
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # İyileştirilmiş heatmap işleme
    cam_enhanced = enhance_heatmap(cam_resized, percentile_low=80.0, percentile_high=99.5, top_percent=25.0, gaussian_blur=5)
    
    # Bbox odaklı zayıflatma (bbox dışındaki heatmap'i bastır)
    if boxes and len(boxes) > 0:
        # Bbox maskesi oluştur
        bbox_mask = np.zeros((h, w), dtype=np.float32)
        for (x1, y1, x2, y2) in boxes:
            # Bbox içini 1.0, dışını 0.0 yap
            bbox_mask[y1:y2, x1:x2] = 1.0
        
        # Bbox içi ve dışı için farklı ağırlıklar
        # Bbox içi: tam güç, bbox dışı: zayıflatılmış
        cam_focused = cam_enhanced.copy()
        cam_focused = cam_focused * (bbox_mask + (1 - bbox_mask) * bbox_focus_strength)
        
        # Dinamik alpha: bbox içi yüksek, dışı düşük
        alpha_map = np.ones((h, w), dtype=np.float32) * alpha_base
        alpha_map = alpha_map + bbox_mask * (alpha_bbox - alpha_base)
    else:
        cam_focused = cam_enhanced
        alpha_map = np.ones((h, w), dtype=np.float32) * alpha_base
    
    # Normalize et (0-255 arası)
    cam_enhanced = (np.clip(cam_focused, 0, 1) * 255).astype(np.uint8)
    
    # TURBO colormap uygula
    heatmap = cv2.applyColorMap(cam_enhanced, cv2.COLORMAP_TURBO)
    
    # Orijinal görüntüyü gri tonlara çevir
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Dinamik alpha blending
    alpha_3d = np.stack([alpha_map] * 3, axis=2)  # [H, W, 3]
    out = (gray_bgr.astype(np.float32) * (1 - alpha_3d) + 
           heatmap.astype(np.float32) * alpha_3d).astype(np.uint8)
    
    return out


def overlay_heatmap_bbox(
    bgr_img: np.ndarray,
    cam: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 20,
    alpha: float = 0.75
) -> np.ndarray:
    """
    Sadece bbox crop üzerine yüksek kontrastlı heatmap overlay.
    
    Args:
        bgr_img: BGR formatında orijinal görüntü
        cam: Normalize edilmiş CAM (0-1 arası, [H, W])
        bbox: (x1, y1, x2, y2) bbox koordinatları
        padding: Bbox etrafına eklenecek padding (piksel, varsayılan: 20)
        alpha: Overlay şeffaflığı (varsayılan: 0.75, yüksek kontrast için)
        
    Returns:
        Bbox crop + overlay edilmiş BGR görüntü
    """
    h, w = bgr_img.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Padding ekle (görüntü sınırları içinde)
    x1_crop = max(0, x1 - padding)
    y1_crop = max(0, y1 - padding)
    x2_crop = min(w, x2 + padding)
    y2_crop = min(h, y2 + padding)
    
    # Crop görüntü
    img_crop = bgr_img[y1_crop:y2_crop, x1_crop:x2_crop].copy()
    h_crop, w_crop = img_crop.shape[:2]
    
    if h_crop == 0 or w_crop == 0:
        return bgr_img  # Geçersiz crop
    
    # CAM'i crop boyutuna ölçekle
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_crop = cam_resized[y1_crop:y2_crop, x1_crop:x2_crop]
    
    # İyileştirilmiş heatmap işleme (daha agresif parametreler)
    cam_enhanced = enhance_heatmap(
        cam_crop,
        percentile_low=75.0,  # Daha agresif
        percentile_high=99.8,
        top_percent=20.0,  # Sadece top %20
        gaussian_blur=3  # Daha az blur (daha keskin)
    )
    
    # Normalize et (0-255 arası)
    cam_enhanced = (np.clip(cam_enhanced, 0, 1) * 255).astype(np.uint8)
    
    # HOT colormap uygula (bbox crop için daha kontrastlı)
    heatmap = cv2.applyColorMap(cam_enhanced, cv2.COLORMAP_HOT)
    
    # Orijinal görüntüyü gri tonlara çevir
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Yüksek kontrastlı overlay
    out_crop = cv2.addWeighted(gray_bgr, 1 - alpha, heatmap, alpha, 0)
    
    # Orijinal görüntüye geri yerleştir
    out = bgr_img.copy()
    out[y1_crop:y2_crop, x1_crop:x2_crop] = out_crop
    
    return out


def draw_detections(
    img: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    confidences: List[float],
    class_names: List[str],
    color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2
) -> np.ndarray:
    """
    Tespit edilen nesneleri görüntü üzerine çizer.
    
    Args:
        img: BGR görüntü
        boxes: [(x1, y1, x2, y2), ...] listesi
        confidences: Confidence değerleri listesi
        class_names: Sınıf isimleri listesi
        color: Bbox rengi (BGR)
        line_thickness: Çizgi kalınlığı
        
    Returns:
        Çizilmiş görüntü
    """
    img_copy = img.copy()
    for (x1, y1, x2, y2), conf, cls_name in zip(boxes, confidences, class_names):
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, line_thickness)
        label = f"{cls_name} {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            img_copy,
            (x1, y1 - label_h - 10),
            (x1 + label_w, y1),
            color,
            -1
        )
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    return img_copy


def save_heatmap_only(
    cam: np.ndarray,
    output_path: str,
    original_shape: Tuple[int, int]
) -> None:
    """
    Sadece heatmap'i kaydeder (bbox olmadan).
    İyileştirilmiş görselleştirme ile.
    
    Args:
        cam: Normalize edilmiş CAM (0-1 arası)
        output_path: Çıktı dosya yolu
        original_shape: (height, width) orijinal görüntü boyutu
    """
    h, w = original_shape
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # İyileştirilmiş heatmap işleme
    cam_enhanced = enhance_heatmap(cam_resized, percentile_low=80.0, percentile_high=99.5, top_percent=25.0, gaussian_blur=5)
    
    # Normalize et
    cam_enhanced = (np.clip(cam_enhanced, 0, 1) * 255).astype(np.uint8)
    
    # TURBO colormap uygula
    heatmap = cv2.applyColorMap(cam_enhanced, cv2.COLORMAP_TURBO)
    cv2.imwrite(output_path, heatmap)


def save_metadata(
    output_path: str,
    detections: List[Dict],
    conf_threshold: float
) -> None:
    """
    Tespit metadata'sını JSON olarak kaydeder.
    
    Args:
        output_path: Çıktı JSON dosya yolu
        detections: Tespit bilgileri listesi
        conf_threshold: Kullanılan confidence threshold
    """
    metadata = {
        "conf_threshold": conf_threshold,
        "num_detections": len(detections),
        "detections": detections
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def format_detection_info(
    x1: int, y1: int, x2: int, y2: int,
    conf: float, cls: int, cls_name: str
) -> Dict:
    """Tespit bilgisini dictionary formatına çevirir."""
    return {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "confidence": float(conf),
        "class_id": int(cls),
        "class_name": str(cls_name)
    }

