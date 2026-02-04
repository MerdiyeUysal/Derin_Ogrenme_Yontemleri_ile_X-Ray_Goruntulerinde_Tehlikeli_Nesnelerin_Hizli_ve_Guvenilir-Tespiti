"""
Grad-CAM implementasyonu YOLOv8 için.
Çekirdek mantık: son Conv2d katmanından gradient toplama ve CAM üretimi.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def find_last_conv(module: nn.Module, min_size: int = 40) -> nn.Module:
    """
    Model içindeki en son uygun Conv2d katmanını bulur (Grad-CAM için).
    Çok küçük feature map'lerden kaçınır.
    
    Args:
        module: PyTorch model modülü
        min_size: Minimum feature map boyutu (varsayılan: 40)
        
    Returns:
        En son uygun Conv2d katmanı
        
    Raises:
        RuntimeError: Uygun Conv2d katmanı bulunamazsa
    """
    # Tüm Conv2d katmanlarını topla
    conv_layers = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_layers.append((name, m))
    
    if not conv_layers:
        raise RuntimeError("Conv2d katmanı bulunamadı. Grad-CAM için conv lazım.")
    
    # En son uygun katmanı bul (feature map boyutu yeterli olan)
    # Genellikle son birkaç katman daha iyi çalışır
    # Son 3-5 katmandan birini seç
    candidates = conv_layers[-min(5, len(conv_layers)):]
    
    # Eğer tek katman varsa onu kullan
    if len(candidates) == 1:
        return candidates[0][1]
    
    # En son uygun katmanı seç (genellikle son 2-3 katman iyidir)
    selected = candidates[-2] if len(candidates) >= 2 else candidates[-1]
    print(f"Grad-CAM target layer seçildi: {selected[0]}")
    return selected[1]


class GradCAM:
    """
    Grad-CAM implementasyonu.
    Forward hook ile aktivasyonları, backward hook ile gradientleri yakalar.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch model
            target_layer: Grad-CAM'in uygulanacağı Conv2d katmanı
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        """Forward ve backward hook'larını kaydeder."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> np.ndarray:
        """
        CAM'i normalize eder (0-1 arası).
        Daha agresif percentile clipping ile iyileştirilmiş kontrast.
        
        Args:
            cam: Ham CAM tensörü [H, W]
            
        Returns:
            Normalize edilmiş CAM numpy array
        """
        cam_np = cam.cpu().numpy()
        
        # Min-max normalize
        cam_min = cam_np.min()
        cam_max = cam_np.max()
        
        if cam_max - cam_min < 1e-8:
            # Eğer tüm değerler aynıysa, uniform dağıt
            print("WARNING: CAM değerleri uniform, gradient'ler düzgün hesaplanmamış olabilir")
            return np.ones_like(cam_np) * 0.5
        
        cam_norm = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)
        
        # Daha agresif percentile clipping (80-99.5 arası)
        p80 = np.percentile(cam_norm, 80)
        p995 = np.percentile(cam_norm, 99.5)
        
        if p995 > p80 and p995 > 0.1:  # Eğer yeterince varyans varsa
            # Percentile clipping uygula
            cam_norm = np.clip(cam_norm, p80, p995)
            # Yeniden normalize et
            cam_norm = (cam_norm - p80) / (p995 - p80 + 1e-8)
        
        # Daha agresif gamma düzeltmesi (kontrastı artır)
        cam_norm = np.power(np.clip(cam_norm, 0, 1), 0.7)
        
        return cam_norm

    def __call__(self, score: torch.Tensor) -> np.ndarray:
        """
        Grad-CAM'i hesaplar.
        
        Args:
            score: Tek skalar tensör (backward için)
            
        Returns:
            Normalize edilmiş CAM [H, W] numpy array
            
        Raises:
            RuntimeError: Hook'lar aktivasyon/grad yakalayamazsa
        """
        self.model.zero_grad(set_to_none=True)
        
        # Backward pass
        score.backward(retain_graph=False)

        # activations: [B, C, H, W]
        # gradients:    [B, C, H, W]
        acts = self.activations
        grads = self.gradients
        
        if acts is None or grads is None:
            raise RuntimeError(
                "Hook'lar aktivasyon/grad yakalayamadı. "
                "Katman seçimi uyumsuz olabilir veya backward çalışmadı."
            )
        
        # Debug: gradient ve activation boyutlarını kontrol et (opsiyonel)
        # print(f"DEBUG: Activations shape: {acts.shape}, Gradients shape: {grads.shape}")
        # print(f"DEBUG: Gradients min/max: {grads.min().item():.6f}/{grads.max().item():.6f}")
        # print(f"DEBUG: Activations min/max: {acts.min().item():.6f}/{acts.max().item():.6f}")

        # Gradient'leri normalize et (çok küçük olabilirler)
        # Global average pooling: her channel için gradient ortalaması
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Gradient'lerin mutlak değerini al (önemli olan büyüklük)
        weights = weights.abs()
        
        # Normalize et (channel'lar arası)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum: channel ağırlıklı aktivasyonlar
        cam = (weights * acts).sum(dim=1, keepdim=False)  # [B, H, W]
        
        # ReLU: sadece pozitif değerler
        cam = torch.relu(cam)[0]  # [H, W]
        
        # print(f"DEBUG: CAM min/max before normalize: {cam.min().item():.6f}/{cam.max().item():.6f}")
        
        return self._normalize_cam(cam)


def compute_differentiable_score(
    core_model: nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    target_bbox: tuple = None
) -> torch.Tensor:
    """
    Core model üzerinden differentiable bir skor üretir.
    Bu skor backward için kullanılır.
    
    Args:
        core_model: YOLO'nun core model'i (yolo.model.model)
        img_tensor: Preprocess edilmiş görüntü tensörü [1, C, H, W]
        device: Cihaz (cuda/cpu)
        
    Returns:
        Skalar skor tensörü (requires_grad=True)
        
    Raises:
        RuntimeError: Model forward çıktısından tensör çekilemezse
    """
    img_t = img_tensor.to(device)
    img_t.requires_grad_(True)
    
    # Core model'i train moduna al (ama BN'leri eval'da tut)
    core_model.train()
    # BatchNorm'ları eval moduna al (stabilite için)
    # Inplace operasyonları devre dışı bırak (gradient sorunlarını önlemek için)
    for m in core_model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
        # Activation modüllerinin inplace operasyonlarını devre dışı bırak
        if hasattr(m, 'inplace'):
            m.inplace = False
    
    # Forward pass - YOLO model'inin forward metodunu güvenli şekilde çağır
    # YOLO model'inin forward metodu genellikle tuple/list döner
    # Ancak bazı durumlarda model içinde Concat gibi modüller farklı input bekleyebilir
    # Çözüm: Model'in sadece backbone kısmını kullan (Concat modüllerinden önce)
    
    with torch.enable_grad():
        # YOLO model yapısı: core_model genellikle nn.Sequential
        # Concat modülleri genellikle head kısmında, backbone'da değil
        # Bu yüzden model'in sadece backbone kısmını kullanıyoruz
        
        # Model yapısını kontrol et
        if hasattr(core_model, 'backbone'):
            # Backbone varsa onu kullan
            backbone = core_model.backbone
            backbone.train()
            for m in backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
            out = backbone(img_t)
        elif isinstance(core_model, nn.Sequential):
            # Sequential model ise, Concat modüllerinden önceki kısmı kullan
            # YOLO'da genellikle ilk 20-30 katman backbone'dur ve Concat içermez
            layers = list(core_model.children())
            
            # Concat modüllerini bul ve onlardan önceki kısmı al
            from ultralytics.nn.modules import Concat
            backbone_layers = []
            for layer in layers:
                if isinstance(layer, Concat):
                    # Concat modülüne gelince dur
                    break
                backbone_layers.append(layer)
            
            # Eğer hiç layer bulunamadıysa, ilk 20 katmanı kullan
            if not backbone_layers:
                backbone_layers = layers[:20] if len(layers) > 20 else layers
            
            if backbone_layers:
                backbone = nn.Sequential(*backbone_layers)
                backbone.to(img_t.device)
                backbone.train()
                for m in backbone.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.eval()
                out = backbone(img_t)
            else:
                # Fallback: sadece ilk katmanı kullan
                if len(layers) > 0:
                    out = layers[0](img_t)
                else:
                    raise RuntimeError("Model'de kullanılabilir katman bulunamadı")
        else:
            # Diğer durumlar için normal forward
            out = core_model(img_t)
        
        # Çıktıyı liste/tuple formatına getir
        if not isinstance(out, (list, tuple)):
            out = [out]
    
    # Çıktıyı skalar skora dönüştür
    # Inplace operasyon sorunlarını önlemek için clone() kullan
    if isinstance(out, (list, tuple)):
        score_tensors = []
        for o in out:
            if torch.is_tensor(o):
                # Inplace operasyon sorunlarını önlemek için clone
                score_tensors.append(o.clone())
        
        if not score_tensors:
            raise RuntimeError("Model forward çıktısından tensör çekilemedi (grad için).")
        
        # Tespit edilen nesneye odaklanmak için feature map'in ilgili bölgesini vurgula
        if target_bbox is not None:
            # Bbox koordinatlarını feature map boyutuna ölçekle
            # img_t: [1, C, H, W] (örn: [1, 3, 640, 640])
            # Feature map genellikle daha küçük (örn: [1, C, 80, 80])
            img_h, img_w = img_t.shape[2], img_t.shape[3]
            x1, y1, x2, y2 = target_bbox
            
            # Her feature map için bbox bölgesini vurgula
            scores = []
            for t in score_tensors:
                # Feature map boyutunu al
                f_h, f_w = t.shape[2], t.shape[3]
                
                # Bbox'ı feature map boyutuna ölçekle
                fx1 = max(0, int(x1 * f_w / img_w))
                fy1 = max(0, int(y1 * f_h / img_h))
                fx2 = min(f_w, int(x2 * f_w / img_w))
                fy2 = min(f_h, int(y2 * f_h / img_h))
                
                # Bbox içindeki değerleri vurgula
                if fx2 > fx1 and fy2 > fy1:
                    bbox_region = t[:, :, fy1:fy2, fx1:fx2]
                    if bbox_region.numel() > 0:
                        # Bbox içindeki değerlerin ortalaması (daha yüksek ağırlık)
                        bbox_score = bbox_region.abs().mean() * 3.0
                        # Tüm feature map'in ortalaması (daha düşük ağırlık)
                        global_score = t.abs().mean() * 0.3
                        scores.append(bbox_score + global_score)
                    else:
                        scores.append(t.abs().mean())
                else:
                    scores.append(t.abs().mean())
            
            score = sum(scores)
        else:
            # Tüm tensörlerden abs().mean() topla
            score = sum(t.abs().mean() for t in score_tensors)
    elif isinstance(out, dict):
        # Dict formatında çıktı
        score_tensors = [v for v in out.values() if torch.is_tensor(v)]
        if not score_tensors:
            raise RuntimeError("Model forward çıktısından tensör çekilemedi (grad için).")
        score = sum(t.abs().mean() for t in score_tensors)
    else:
        if not torch.is_tensor(out):
            raise RuntimeError(f"Model forward çıktısı tensör değil. Tip: {type(out)}")
        score = out.abs().mean()
    
    return score

