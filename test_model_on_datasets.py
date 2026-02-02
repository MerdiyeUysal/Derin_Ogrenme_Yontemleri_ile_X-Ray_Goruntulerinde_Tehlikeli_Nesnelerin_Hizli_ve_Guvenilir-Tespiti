"""Evaluate the fine-tuned YOLOv8 model on multiple datasets.

This script runs the *already trained* two-stage model on:

- the original SIXRay-style dataset, and
- the prohibited items dataset,

and collects metrics in order to analyse potential catastrophic interference.
The training procedure, model weights and YOLOv8 hyperparameters are not
modified here. Only evaluation and reporting logic lives in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from ultralytics import YOLO

from config import PATHS, RUNTIME


def load_data_yaml(yaml_path: Path) -> dict:
    """
    Load a YOLO-style ``data.yaml`` file.

    Parameters
    ----------
    yaml_path:
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content.
    """

    with open(yaml_path, "r", encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file)


def test_model_on_dataset(
    model_path: Path,
    data_yaml_path: Path,
    dataset_name: str,
    output_dir: Path,
) -> dict | None:
    """
    Run model evaluation on a specific dataset and collect summary metrics.

    The underlying YOLOv8 ``val`` logic, including hyperparameters, is kept as
    close as possible to the original project. This function only wraps the
    call, performs basic checks and shapes the returned metrics.

    Parameters
    ----------
    model_path:
        Path to the trained YOLOv8 weights.
    data_yaml_path:
        Path to the dataset's ``data.yaml`` description.
    dataset_name:
        Human-readable dataset name used in logs and report files.
    output_dir:
        Directory where YOLO will write its plots and intermediate results.

    Returns
    -------
    dict | None
        A dictionary with aggregate and per-class metrics, or ``None`` if the
        dataset could not be evaluated (e.g. missing test split).
    """
    print(f"\n{'='*60}")
    print(f"Test Veri Seti: {dataset_name}")
    print(f"{'='*60}")
    
    # Model yükle
    print(f"Model yükleniyor: {model_path}")
    model = YOLO(str(model_path))
    
    # Veri seti bilgilerini yükle
    data_info = load_data_yaml(data_yaml_path)
    print(f"Veri seti: {data_yaml_path}")
    print(f"Sınıf sayısı: {data_info.get('nc', 'N/A')}")
    print(f"Sınıf isimleri: {data_info.get('names', 'N/A')}")
    
    # Test dizinini kontrol et
    test_dir = data_yaml_path.parent / "test"
    if not test_dir.exists():
        print(f"⚠️  Test dizini bulunamadı: {test_dir}")
        return None
    
    # Test görüntülerini say
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    print(f"Test görüntü sayısı: {len(test_images)}")
    
    # Validation çalıştır (test seti üzerinde)
    print("\nModel test ediliyor...")
    results = model.val(
        data=str(data_yaml_path),
        split='test',  # Test setini kullan
        imgsz=640,
        conf=0.25,
        iou=0.7,
        save_json=False,
        plots=True,
        project=str(output_dir),
        name=f"test_{dataset_name.lower().replace(' ', '_')}",
        verbose=True
    )
    
    # Metrikleri topla
    import numpy as np
    
    def safe_float(value):
        """Convert value to float, handling arrays."""
        if value is None:
            return None
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) > 0:
                return float(value[0] if len(value) == 1 else np.mean(value))
            return None
        return float(value)
    
    metrics = {
        'dataset_name': dataset_name,
        'model_path': str(model_path),
        'data_yaml': str(data_yaml_path),
        'test_images_count': len(test_images),
        'metrics': {
            'precision': safe_float(getattr(results.box, 'p', None)),
            'recall': safe_float(getattr(results.box, 'r', None)),
            'mAP50': safe_float(getattr(results.box, 'map50', None)),
            'mAP50_95': safe_float(getattr(results.box, 'map', None)),
            'f1': safe_float(getattr(results.box, 'f1', None)),
        },
        'class_metrics': {}
    }
    
    # Sınıf bazlı metrikler (eğer varsa)
    if hasattr(results.box, 'maps'):
        maps = results.box.maps
        if maps is not None and len(maps) > 0:
            class_names = data_info.get('names', [])
            for i, map_val in enumerate(maps):
                class_name = class_names[i] if i < len(class_names) else f"class_{i}"
                metrics['class_metrics'][class_name] = float(map_val)
    
    # Metrikleri yazdır
    print(f"\n{'='*60}")
    print(f"TEST SONUÇLARI - {dataset_name}")
    print(f"{'='*60}")
    print(f"Precision:  {metrics['metrics']['precision']:.4f}" if metrics['metrics']['precision'] else "Precision:  N/A")
    print(f"Recall:     {metrics['metrics']['recall']:.4f}" if metrics['metrics']['recall'] else "Recall:     N/A")
    print(f"mAP50:      {metrics['metrics']['mAP50']:.4f}" if metrics['metrics']['mAP50'] else "mAP50:      N/A")
    print(f"mAP50-95:   {metrics['metrics']['mAP50_95']:.4f}" if metrics['metrics']['mAP50_95'] else "mAP50-95:   N/A")
    print(f"F1-Score:   {metrics['metrics']['f1']:.4f}" if metrics['metrics']['f1'] else "F1-Score:   N/A")
    
    if metrics['class_metrics']:
        print(f"\nSınıf Bazlı mAP50:")
        for class_name, map_val in metrics['class_metrics'].items():
            print(f"  {class_name}: {map_val:.4f}")
    
    return metrics


def main() -> None:
    """
    Entry point for catastrophic interference evaluation.

    This function wires the centrally configured paths from ``config.py`` to
    the evaluation helper, aggregates metrics from all datasets and prints a
    concise comparison summary to stdout.
    """

    model_path = PATHS.finetuned_weights

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Project root: {PATHS.project_root}")
        return

    datasets = [
        {
            "name": "Sixray",
            "yaml": PATHS.sixray_data_yaml,
        },
        {
            "name": "Prohibited Items",
            "yaml": PATHS.prohibited_items_data_yaml,
        },
    ]

    output_dir = PATHS.catastrophic_eval_dir
    output_dir.mkdir(exist_ok=True)
    
    # Her iki veri seti için test yap
    all_metrics = {}
    
    for dataset in datasets:
        if not dataset['yaml'].exists():
            print(f"⚠️  Veri seti yaml dosyası bulunamadı: {dataset['yaml']}")
            continue
        
        metrics = test_model_on_dataset(
            model_path=model_path,
            data_yaml_path=dataset['yaml'],
            dataset_name=dataset['name'],
            output_dir=output_dir
        )
        
        if metrics:
            all_metrics[dataset['name']] = metrics
    
    # Sonuçları JSON olarak kaydet
    results_file = output_dir / "test_metrics_comparison.json"
    with open(results_file, "w", encoding="utf-8") as results_fp:
        json.dump(all_metrics, results_fp, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("KARŞILAŞTIRMA ÖZETİ")
    print(f"{'='*60}")
    
    if len(all_metrics) == 2:
        sixray = all_metrics.get('Sixray', {})
        prohibited = all_metrics.get('Prohibited Items', {})
        
        print(f"\n{'Metrik':<20} {'Sixray':<15} {'Prohibited Items':<15} {'Fark':<15}")
        print("-" * 65)
        
        for metric_key in ['precision', 'recall', 'mAP50', 'mAP50_95', 'f1']:
            sixray_val = sixray.get('metrics', {}).get(metric_key)
            prohibited_val = prohibited.get('metrics', {}).get(metric_key)
            
            if sixray_val is not None and prohibited_val is not None:
                diff = prohibited_val - sixray_val
                diff_pct = (diff / sixray_val * 100) if sixray_val > 0 else 0
                print(f"{metric_key:<20} {sixray_val:<15.4f} {prohibited_val:<15.4f} {diff:+.4f} ({diff_pct:+.2f}%)")
            else:
                print(f"{metric_key:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        # Catastrophic interference analizi
        print(f"\n{'='*60}")
        print("CATASTROPHIC INTERFERENCE ANALİZİ")
        print(f"{'='*60}")
        
        if sixray.get('metrics', {}).get('mAP50') and prohibited.get('metrics', {}).get('mAP50'):
            sixray_map = sixray['metrics']['mAP50']
            prohibited_map = prohibited['metrics']['mAP50']
            
            # Eğer Prohibited Items'da başarılı ama Sixray'de düşükse, interference var demektir
            if prohibited_map > 0.80 and sixray_map < 0.70:
                print("⚠️  CATASTROPHIC INTERFERENCE TESPİT EDİLDİ!")
                print(f"   - Prohibited Items mAP50: {prohibited_map:.4f} (İyi)")
                print(f"   - Sixray mAP50: {sixray_map:.4f} (Düşük)")
                print(f"   - Model fine-tuning sırasında ilk veri setindeki bilgileri unutmuş olabilir.")
            elif prohibited_map > 0.80 and sixray_map > 0.70:
                print("✅ Catastrophic interference görünmüyor.")
                print(f"   - Her iki veri setinde de model iyi performans gösteriyor.")
            else:
                print("ℹ️  Model her iki veri setinde de düşük performans gösteriyor.")
                print(f"   - Bu durum interference'dan ziyade genel model performans sorunu olabilir.")
    
    print(f"\n✅ Tüm sonuçlar kaydedildi: {results_file}")
    print(f"✅ Test çıktıları: {output_dir}")


if __name__ == "__main__":
    main()

