"""RapidOCR 文字识别服务（中文印刷体，CPU，懒加载单例）。"""
import threading
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageEnhance

_engine = None
_lock = threading.Lock()

# 分块识别参数：RapidOCR 检测模型内部会把最长边缩到 ~736px，
# 整图直接识别会丢失包装小字（尤其红色低对比度的生产日期行）。
# 策略：最长边超过 _TILE 就切块（每块原尺寸进检测器），
# 每块再附加一个"2倍放大+对比度增强"变体救小字，结果按文本去重合并。
_TILE = 736            # 每块边长
_OVERLAP = 120         # 块间重叠，避免文字被切断
_HI_RES_MIN_SIDE = 1280 # 最长边超过此值才分块（原 736 过激：手机照片普遍
                          # 1000~2000，几乎全被分块；RapidOCR 内部总会把最长边
                          # 缩到 ~736 识别，分块并非必要，反而倍增 OCR 调用数）
_UPSCALE = 2           # 附加放大倍数
_UPSCALE_CONTRAST = 1.8
_UPSCALE_MAX_SIDE = 1500  # 仅对较小图启用放大变体（大图靠分块已够，控性能）
_MAX_OCR_SIDE = 1600     # 原图最长边超过此值先整体降采样再分块：
                         # RapidOCR 内部总把最长边缩到 ~736 识别，超大原图只增
                         # 分块数与放大开销、不增可识别信息；降采样大幅提速且
                         # 文字最终识别分辨率不变（坐标最后映射回原图）。


def get_engine():
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                from rapidocr_onnxruntime import RapidOCR
                _engine = RapidOCR()
    return _engine


def ocr_ready() -> bool:
    try:
        get_engine()
        return True
    except Exception:
        return False


def _ocr_array(engine, pil_image: Image.Image) -> List[Dict]:
    """对单块图像执行 OCR，返回统一结构。"""
    arr = np.asarray(pil_image.convert("RGB"))
    result, _ = engine(arr)
    out: List[Dict] = []
    if not result:
        return out
    for item in result:
        box, text, score = item
        out.append({
            "box": [[float(x), float(y)] for x, y in box],
            "text": str(text),
            "score": round(float(score), 3),
        })
    return out


def _tiles(w: int, h: int):
    """把 w×h 切成带重叠的块列表 [(x, y, tw, th), ...]。"""
    step = _TILE - _OVERLAP
    cols = max(1, -(-(w - _OVERLAP) // step)) if w > _TILE else 1
    rows = max(1, -(-(h - _OVERLAP) // step)) if h > _TILE else 1
    out = []
    for r in range(rows):
        for c in range(cols):
            x = min(c * step, max(0, w - _TILE)) if w > _TILE else 0
            y = min(r * step, max(0, h - _TILE)) if h > _TILE else 0
            tw, th = min(_TILE, w - x), min(_TILE, h - y)
            out.append((x, y, tw, th))
    return out


def _ocr_upscaled(engine, pil_image: Image.Image) -> List[Dict]:
    """2 倍放大 + 对比度增强的补充识别（坐标已映射回原图）。

    放大使小字变大，再整体送 RapidOCR（其内部会把最长边缩到 ~736 识别），
    单次即可——**不再对放大图二次分块**。二次分块会把一次补充识别拆成 6~9 次
    OCR 调用，是此前单张 OCR 耗时高达 4s+ 的主因。放大变体仅在原图最长边
    <= _UPSCALE_MAX_SIDE 时启用（见 ocr()），故放大后最长边 <= 3000，单次
    RapidOCR 完全能承受，且能读回原分辨率漏掉的低对比度小字。
    """
    w, h = pil_image.size
    up = pil_image.resize((w * _UPSCALE, h * _UPSCALE), Image.LANCZOS)
    if up.mode != "RGB":                    # ImageEnhance.Contrast 只吃 RGB
        up = up.convert("RGB")
    up = ImageEnhance.Contrast(up).enhance(_UPSCALE_CONTRAST)
    out: List[Dict] = []
    for item in _ocr_array(engine, up):
        item["box"] = [[int(round(x / _UPSCALE)), int(round(y / _UPSCALE))]
                       for x, y in item["box"]]
        out.append(item)
    return out


def _dedup(items: List[Dict]) -> List[Dict]:
    """同文本保留高置信版本。"""
    best: Dict[str, Dict] = {}
    for it in items:
        t = it["text"]
        if t not in best or best[t]["score"] < it["score"]:
            best[t] = it
    return list(best.values())


def _need_upscale(items: List[Dict], min_lines: int = 5,
                  min_avg_conf: float = 0.75) -> bool:
    """判断是否需要做 2 倍放大增强 OCR。

    放大增强是为救「低对比度/小字号」文字（如红色生产日期行），但代价是
    再跑一遍 OCR。当原图/原块已清晰读到足够多文字时，放大纯属冗余——
    直接跳过可砍掉整段补充识别耗时，且不损失可用文字（日期另有点阵兜底）。

    返回 True 表示需要放大（文字过少或整体模糊），False 表示原图已够清晰。
    """
    if not items:
        return True                      # 没读到任何字，必须放大尝试救字
    if len(items) >= min_lines:
        return False                     # 已读到足够多文字，放大救字价值低，跳过提速
    if len(items) < 3:
        return True                      # 文字太少，放大尝试救字
    avg = sum(i["score"] for i in items) / len(items)
    if avg < min_avg_conf:
        return True                      # 整体模糊，放大救
    return False


def ocr(pil_image: Image.Image) -> List[Dict]:
    """
    整图 OCR。返回 [{box:[[x,y]x4], text, score}]，坐标为原图坐标。
    大图自动分块识别以保住包装小字（重叠区域按文本去重、保留高置信分），
    每块附加放大增强变体救回低对比度小字（如红色生产日期行）。

    提速：原图最长边超过 _MAX_OCR_SIDE 时先整体降采样——RapidOCR 内部检测
    总会把最长边缩到 ~736 识别，原图超过此值只会增加分块数与放大开销，
    不增加可识别信息；降采样大幅减少大图 OCR 耗时，且文字最终识别分辨率
    不变。最后把所有 box 坐标乘回原图比例，保证后续 texts_inside 等坐标一致。
    """
    engine = get_engine()
    ow, oh = pil_image.size
    scale = 1.0
    if max(ow, oh) > _MAX_OCR_SIDE:
        s = _MAX_OCR_SIDE / max(ow, oh)
        pil_image = pil_image.resize((int(ow * s), int(oh * s)), Image.LANCZOS)
        scale = 1.0 / s                       # 结果坐标需乘回原图
    w, h = pil_image.size
    variants_on = max(w, h) <= _UPSCALE_MAX_SIDE
    if max(w, h) <= _HI_RES_MIN_SIDE:
        items = _ocr_array(engine, pil_image)
        if variants_on and _need_upscale(items):
            items += _ocr_upscaled(engine, pil_image)
        results = _dedup(items)
    else:
        best: Dict[str, Dict] = {}
        for (tx, ty, tw, th) in _tiles(w, h):
            crop = pil_image.crop((tx, ty, tx + tw, ty + th))
            items = _ocr_array(engine, crop)
            if variants_on and _need_upscale(items):
                items += _ocr_upscaled(engine, crop)
            for item in items:
                item["box"] = [[x + tx, y + ty] for x, y in item["box"]]
                text = item["text"]
                if text not in best or best[text]["score"] < item["score"]:
                    best[text] = item
        results = list(best.values())
    if scale != 1.0:
        for it in results:
            it["box"] = [[x * scale, y * scale] for x, y in it["box"]]
    return results


def ocr_texts(pil_image: Image.Image) -> List[str]:
    return [t["text"] for t in ocr(pil_image)]


def texts_inside(pil_image: Image.Image, bbox: List[int], pad: int = 0,
                 items: Optional[List[Dict]] = None) -> List[str]:
    """找出中心点落在 bbox（外扩 pad 像素）内的 OCR 文本行。

    items 可传入已有的 OCR 结果（含点阵喷码兜底文本），避免重复识别。
    """
    if items is None:
        items = ocr(pil_image)
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    out = []
    for t in items:
        xs = [p[0] for p in t["box"]]
        ys = [p[1] for p in t["box"]]
        cx, cy = sum(xs) / 4, sum(ys) / 4
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            out.append(t["text"])
    return out
