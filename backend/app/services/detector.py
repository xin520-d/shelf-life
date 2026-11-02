"""YOLO11n 目标检测服务（CPU 推理，懒加载单例）。"""
import threading
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

from .. import config

_model = None
_lock = threading.Lock()


def get_model():
    """懒加载 YOLO 模型；权重不存在时由 ultralytics 自动下载。"""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from ultralytics import YOLO
                weights = config.MODEL_DIR / config.YOLO_MODEL
                if weights.exists():
                    _model = YOLO(str(weights))
                else:
                    _model = YOLO(config.YOLO_MODEL)  # 触发 GitHub 下载
    return _model


def model_ready() -> bool:
    try:
        get_model()
        return True
    except Exception:
        return False


def _resize(pil: Image.Image, max_side: int) -> Image.Image:
    w, h = pil.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return pil
    return pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def detect(pil_image: Image.Image) -> List[Dict]:
    """
    对整张照片做多目标检测。
    返回 [{class_name, confidence, bbox:[x1,y1,x2,y2]}]，bbox 为原图坐标。
    只保留 COCO_ITEM_CLASSES 白名单内的类别（过滤 person/car 等）。
    """
    model = get_model()
    original_size = pil_image.size
    img = _resize(pil_image, config.MAX_IMAGE_SIZE)
    results = model.predict(
        source=np.asarray(img.convert("RGB")),
        conf=config.DETECT_CONF,
        iou=config.DETECT_IOU,
        verbose=False,
        device="cpu",
    )
    detections: List[Dict] = []
    if not results:
        return detections
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return detections
    # 缩放回原图坐标
    sx = original_size[0] / img.size[0]
    sy = original_size[1] / img.size[1]
    for b in boxes:
        cls_id = int(b.cls[0])
        cls_name = model.names.get(cls_id, str(cls_id))
        if cls_name not in config.COCO_ITEM_CLASSES:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        detections.append({
            "class_name": cls_name,
            "confidence": round(float(b.conf[0]), 3),
            "bbox": [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections
