"""
YOLOv8 Grad-CAM: Bbox-focused, object-isolated XAI for academic presentation.
Restricts CAM to detected bounding box, suppresses background, emphasizes object.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, List, Optional
from ultralytics import YOLO


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Ultralytics letterbox preprocessing."""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (ratio, (dw, dh))


def find_last_conv_in_backbone(model: nn.Module) -> Optional[nn.Module]:
    """Find the last Conv2d layer in the backbone."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def select_best_bbox(boxes_xyxy: np.ndarray, confidences: np.ndarray, class_ids: np.ndarray, 
                     min_area: int = 100, min_conf: float = 0.25) -> Tuple[int, np.ndarray, float, int]:
    """Select the best bbox: prefer larger boxes with meaningful area."""
    if len(boxes_xyxy) == 0:
        return None, None, None, None
    
    valid_indices = []
    for i, (box, conf) in enumerate(zip(boxes_xyxy, confidences)):
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area:
            valid_indices.append(i)
    
    if not valid_indices:
        best_idx = int(np.argmax(confidences))
        return best_idx, boxes_xyxy[best_idx], confidences[best_idx], class_ids[best_idx]
    
    scores = []
    for idx in valid_indices:
        x1, y1, x2, y2 = boxes_xyxy[idx]
        area = (x2 - x1) * (y2 - y1)
        score = area * confidences[idx]
        scores.append((score, idx))
    
    scores.sort(reverse=True)
    best_idx = scores[0][1]
    return best_idx, boxes_xyxy[best_idx], confidences[best_idx], class_ids[best_idx]


def create_bbox_mask_in_feature_space(bbox_xyxy: np.ndarray, feature_shape: Tuple[int, int], 
                                       img_shape: Tuple[int, int], letterbox_info: Tuple) -> torch.Tensor:
    """
    Create a binary mask in feature map space corresponding to the bbox.
    Returns mask: [1, H_feat, W_feat] with 1.0 inside bbox, 0.0 outside.
    """
    x1, y1, x2, y2 = bbox_xyxy
    feat_h, feat_w = feature_shape
    img_h, img_w = img_shape
    
    # Map bbox from original image to letterbox coordinates
    ratio, (dw, dh) = letterbox_info
    ratio_w, ratio_h = ratio
    
    # Bbox in letterbox space
    x1_lb = x1 * ratio_w + dw
    y1_lb = y1 * ratio_h + dh
    x2_lb = x2 * ratio_w + dw
    y2_lb = y2 * ratio_h + dh
    
    # Map to feature map space (assuming 640x640 input -> feature map size)
    # Feature map is typically 1/32 of input size for last backbone layer
    scale_factor_h = feat_h / 640.0
    scale_factor_w = feat_w / 640.0
    
    fx1 = max(0, int(x1_lb * scale_factor_w))
    fy1 = max(0, int(y1_lb * scale_factor_h))
    fx2 = min(feat_w, int(x2_lb * scale_factor_w))
    fy2 = min(feat_h, int(y2_lb * scale_factor_h))
    
    # Create mask
    mask = torch.zeros(1, feat_h, feat_w)
    if fx2 > fx1 and fy2 > fy1:
        mask[0, fy1:fy2, fx1:fx2] = 1.0
    
    return mask


class BboxFocusedGradCAM:
    """
    Grad-CAM restricted to bounding box region.
    Suppresses background, emphasizes object.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def compute_cam_with_mask(self, bbox_mask: torch.Tensor) -> np.ndarray:
        """
        Compute Grad-CAM with bbox mask applied.
        CAM outside bbox is suppressed, inside is emphasized.
        """
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured.")
        
        # Resize mask to match feature map size
        feat_h, feat_w = self.activations.shape[2:]
        mask_h, mask_w = bbox_mask.shape[1:]
        
        if (mask_h, mask_w) != (feat_h, feat_w):
            bbox_mask_resized = F.interpolate(
                bbox_mask.unsqueeze(0).float(),
                size=(feat_h, feat_w),
                mode='nearest'
            )[0]
        else:
            bbox_mask_resized = bbox_mask
        
        # Global average pooling of gradients (channel-wise weights)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=False)  # [B, H, W]
        cam = F.relu(cam)  # Apply ReLU
        
        # Apply bbox mask: suppress outside, emphasize inside
        cam_masked = cam * bbox_mask_resized  # [B, H, W]
        
        # Further emphasize inside bbox (optional: can increase contrast)
        cam_masked = cam_masked[0].detach().cpu().numpy()  # [H, W]
        
        return cam_masked
    
    def normalize_cam_per_bbox(self, cam: np.ndarray, bbox_mask_img_space: np.ndarray) -> np.ndarray:
        """
        Normalize CAM only within bbox region (not whole image).
        This ensures high contrast for the object.
        """
        # Ensure non-negative
        cam = np.maximum(cam, 0)
        
        # Extract bbox region
        mask_bool = bbox_mask_img_space > 0.5
        if mask_bool.sum() == 0:
            # Fallback to global normalization
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            return cam
        
        # Normalize only within bbox
        cam_bbox = cam[mask_bool]
        if len(cam_bbox) > 0:
            cam_min = cam_bbox.min()
            cam_max = cam_bbox.max()
            
            if cam_max - cam_min > 1e-8:
                # Normalize whole image using bbox statistics
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
                # Suppress outside bbox
                cam = cam * bbox_mask_img_space
            else:
                cam = cam * bbox_mask_img_space
        
        # Light Gaussian smoothing
        cam = cv2.GaussianBlur(cam, (5, 5), 1.0)
        
        # Re-normalize after blur (within bbox)
        cam_bbox = cam[mask_bool]
        if len(cam_bbox) > 0 and cam_bbox.max() - cam_bbox.min() > 1e-8:
            cam_min = cam_bbox.min()
            cam_max = cam_bbox.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            cam = cam * bbox_mask_img_space  # Re-apply mask
        
        return cam


def extract_detection_specific_score(model_output, target_class_id: int, bbox_xyxy: np.ndarray,
                                     img_shape: Tuple[int, int], letterbox_info: Tuple) -> torch.Tensor:
    """
    Extract score tied to the specific detection (not global).
    Uses class logit of detected class + objectness.
    Do NOT use global feature averages.
    """
    if isinstance(model_output, (list, tuple)):
        all_scores = []
        
        for output in model_output:
            if not torch.is_tensor(output):
                continue
            
            # output shape: [batch, num_anchors, 5+num_classes]
            objectness = output[..., 4:5].squeeze(-1)  # [batch, num_anchors]
            class_logits = output[..., 5:]  # [batch, num_anchors, num_classes]
            target_class_logit = class_logits[..., target_class_id]  # [batch, num_anchors]
            
            # Combine: objectness * class_logit (detection-specific)
            combined = objectness * target_class_logit  # [batch, num_anchors]
            all_scores.append(combined)
        
        if not all_scores:
            raise RuntimeError("No valid output tensors found!")
        
        # Take max across all scales and anchors (most confident detection)
        all_scores_flat = [s.flatten() for s in all_scores]
        all_scores_combined = torch.cat(all_scores_flat, dim=0)
        combined_score = all_scores_combined.max()
        
    elif torch.is_tensor(model_output):
        objectness = model_output[..., 4:5].squeeze(-1)
        class_logits = model_output[..., 5:]
        target_class_logit = class_logits[..., target_class_id]
        combined_score = (objectness * target_class_logit).max()
    else:
        raise RuntimeError(f"Unexpected model output type: {type(model_output)}")
    
    return combined_score


def main():
    """Main function."""
    import sys
    
    # === SETTINGS ===
    script_dir = Path(__file__).parent.parent.absolute()
    model_path = script_dir / "runs_finetune/sixray_to_prohibited_finish/weights/finetuned_best.pt"
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if not image_path.is_absolute():
            image_path = script_dir / image_path
    else:
        image_path = script_dir / "xai_input.jpg"
    
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
        if not output_dir.is_absolute():
            output_dir = script_dir / output_dir
    else:
        output_dir = script_dir / "xai_outputs"
    
    conf_threshold = 0.25
    imgsz = 640
    
    # === PATH VALIDATION ===
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"MODEL: {model_path.absolute()}")
    print(f"IMAGE: {image_path.absolute()}")
    print(f"OUTPUT: {Path(output_dir).absolute()}")
    
    # === LOAD MODEL ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")
    
    yolo = YOLO(str(model_path))
    yolo_model = yolo.model
    
    # Extract backbone for finding target layer
    from ultralytics.nn.modules import Concat
    full_model_seq = yolo_model.model
    if isinstance(full_model_seq, nn.Sequential):
        layers = list(full_model_seq.children())
        backbone_layers = []
        for layer in layers:
            if isinstance(layer, Concat):
                break
            backbone_layers.append(layer)
        if backbone_layers:
            backbone = nn.Sequential(*backbone_layers)
        else:
            backbone = nn.Sequential(*layers[:20])
    else:
        backbone = full_model_seq
    
    backbone.to(device)
    yolo_model.to(device)
    
    # Find last Conv2d in backbone
    target_layer = find_last_conv_in_backbone(backbone)
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found in backbone!")
    
    # Find same layer in full model
    target_layer_full = None
    for name, module in yolo_model.named_modules():
        if module is target_layer:
            target_layer_full = module
            break
    
    if target_layer_full is None:
        target_layer_full = find_last_conv_in_backbone(yolo_model)
        if target_layer_full is None:
            raise RuntimeError("Cannot find target layer in full model!")
    
    print(f"Target layer: {target_layer_full}")
    
    # === LOAD AND PREPROCESS IMAGE ===
    img_orig = cv2.imread(str(image_path))
    if img_orig is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    h_orig, w_orig = img_orig.shape[:2]
    print(f"Original size: {w_orig}x{h_orig}")
    
    # Letterbox preprocessing
    img_letterbox, letterbox_info = letterbox(img_orig, new_shape=imgsz, auto=True)
    img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # === GET BBOXES (inference only) ===
    print("Running YOLO inference to get bboxes...")
    with torch.no_grad():
        results = yolo.predict(source=str(image_path), conf=conf_threshold, imgsz=imgsz, verbose=False)
    
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        print(f"No detections (conf={conf_threshold}). Try lower conf threshold.")
        return
    
    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    boxes_xyxy = boxes.xyxy.cpu().numpy()
    
    # Select best bbox
    best_idx, best_bbox, best_conf, best_cls = select_best_bbox(
        boxes_xyxy, confs, class_ids, min_area=100, min_conf=conf_threshold
    )
    
    if best_bbox is None:
        print("No suitable bbox found.")
        return
    
    cls_name = results[0].names.get(best_cls, str(best_cls))
    x1, y1, x2, y2 = best_bbox.astype(int)
    print(f"Selected bbox: {cls_name} conf={best_conf:.3f} ({x1},{y1},{x2},{y2})")
    
    # === GRAD-CAM COMPUTATION ===
    print("Computing bbox-focused Grad-CAM...")
    
    gradcam = BboxFocusedGradCAM(yolo_model, target_layer_full)
    
    # Prepare model for training mode
    yolo_model.train()
    for m in yolo_model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
        if hasattr(m, 'inplace'):
            m.inplace = False
    
    img_tensor.requires_grad_(True)
    
    # Forward pass
    with torch.enable_grad():
        model_output = yolo_model.forward(img_tensor)
    
    # Get feature map shape for mask creation
    if gradcam.activations is not None:
        feat_h, feat_w = gradcam.activations.shape[2:]
    else:
        # Fallback: estimate from typical YOLOv8 structure
        feat_h, feat_w = 20, 20  # Last backbone layer is typically 1/32 of 640
    
    # Create bbox mask in feature space
    bbox_mask_feat = create_bbox_mask_in_feature_space(
        best_bbox, (feat_h, feat_w), (h_orig, w_orig), letterbox_info
    ).to(device)
    
    # Extract detection-specific score (not global)
    score = extract_detection_specific_score(
        model_output, int(best_cls), best_bbox, (h_orig, w_orig), letterbox_info
    )
    print(f"Detection-specific score (objectness * class_logit): {score.item():.4f}")
    
    # Backward pass
    yolo_model.zero_grad(set_to_none=True)
    score.backward()
    
    # Compute CAM with bbox mask
    cam = gradcam.compute_cam_with_mask(bbox_mask_feat)
    print(f"CAM computed, shape: {cam.shape}")
    
    # === MAP CAM TO ORIGINAL IMAGE ===
    # Resize CAM to letterbox size
    cam_letterbox = cv2.resize(cam, (imgsz, imgsz), interpolation=cv2.INTER_CUBIC)
    
    # Inverse letterbox: extract original image region
    ratio, (dw, dh) = letterbox_info
    ratio_w, ratio_h = ratio
    
    pad_w = int(round(dw))
    pad_h = int(round(dh))
    
    orig_w_scaled = int(round(w_orig * ratio_w))
    orig_h_scaled = int(round(h_orig * ratio_h))
    
    if pad_h >= 0 and pad_w >= 0 and pad_h + orig_h_scaled <= imgsz and pad_w + orig_w_scaled <= imgsz:
        cam_cropped = cam_letterbox[pad_h:pad_h+orig_h_scaled, pad_w:pad_w+orig_w_scaled]
    else:
        cam_cropped = cam_letterbox
    
    # Resize to original image size
    if cam_cropped.size > 0:
        cam_resized = cv2.resize(cam_cropped, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    else:
        cam_resized = cv2.resize(cam_letterbox, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    # Normalize CAM for entire image (not just bbox)
    # Ensure non-negative
    cam_resized = np.maximum(cam_resized, 0)
    
    # Normalize entire image
    cam_min = cam_resized.min()
    cam_max = cam_resized.max()
    if cam_max - cam_min > 1e-8:
        cam_normalized = (cam_resized - cam_min) / (cam_max - cam_min + 1e-8)
    else:
        cam_normalized = cam_resized
    
    # Light Gaussian smoothing for better visualization
    cam_normalized = cv2.GaussianBlur(cam_normalized, (5, 5), 1.0)
    
    # Re-normalize after blur
    cam_min = cam_normalized.min()
    cam_max = cam_normalized.max()
    if cam_max - cam_min > 1e-8:
        cam_normalized = (cam_normalized - cam_min) / (cam_max - cam_min + 1e-8)
    
    # === VISUALIZATION ===
    # Heatmap kontrastını artır (daha belirgin olsun) - tüm görüntü için
    cam_for_heatmap = cam_normalized.copy()
    # Gamma correction ile kontrastı artır
    cam_for_heatmap = np.power(cam_for_heatmap, 0.6)  # 0.6 değeri heatmap'i daha belirgin yapar
    # Tekrar normalize et
    if cam_for_heatmap.max() > cam_for_heatmap.min():
        cam_for_heatmap = (cam_for_heatmap - cam_for_heatmap.min()) / (cam_for_heatmap.max() - cam_for_heatmap.min() + 1e-8)
    
    # High-contrast heatmap - TÜM GÖRÜNTÜYE UYGULA
    heatmap = cv2.applyColorMap((cam_for_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Full image overlay - heatmap daha belirgin ama arka plan da görünür
    # 0.60 orijinal görüntü, 0.40 heatmap - daha dengeli ve belirgin
    overlay_full = cv2.addWeighted(img_orig, 0.60, heatmap, 0.40, 0)
    cv2.rectangle(overlay_full, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"{cls_name} {best_conf:.2f}"
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(overlay_full, (x1, y1-label_h-10), (x1+label_w, y1), (0, 255, 0), -1)
    cv2.putText(overlay_full, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Bbox crop (high-contrast, main result)
    bbox_crop_img = img_orig[y1:y2, x1:x2].copy()
    cam_crop = cam_normalized[y1:y2, x1:x2]
    
    # High contrast for bbox crop
    if cam_crop.max() - cam_crop.min() > 1e-8:
        cam_crop_norm = (cam_crop - cam_crop.min()) / (cam_crop.max() - cam_crop.min() + 1e-8)
    else:
        cam_crop_norm = cam_crop
    
    heatmap_crop = cv2.applyColorMap((cam_crop_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay_bbox = cv2.addWeighted(bbox_crop_img, 0.5, heatmap_crop, 0.5, 0)  # Higher contrast
    
    # Draw border
    cv2.rectangle(overlay_bbox, (0, 0), (bbox_crop_img.shape[1]-1, bbox_crop_img.shape[0]-1), (0, 255, 0), 3)
    cv2.putText(overlay_bbox, label, (5, bbox_crop_img.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # === SAVE ===
    cv2.imwrite(str(Path(output_dir) / "xai_overlay_full.jpg"), overlay_full)
    cv2.imwrite(str(Path(output_dir) / "xai_overlay_bbox.jpg"), overlay_bbox)
    
    print(f"✅ Saved: {Path(output_dir) / 'xai_overlay_full.jpg'}")
    print(f"✅ Saved: {Path(output_dir) / 'xai_overlay_bbox.jpg'}")
    print("\n🎉 Bbox-focused Grad-CAM analysis completed!")


if __name__ == "__main__":
    main()
