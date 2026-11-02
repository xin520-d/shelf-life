"""区域 CLIP 多物品扫描（货架/冰箱照片的第四条识别通道）。

为什么需要它：
  - YOLO11n 只认 COCO 80 类，冰箱/货架上的芝士盒、黄油、马苏里拉、
    保鲜膜包装等整类检不出（实测 900x600 冰箱照片 conf=0.35 时 0 框，
    降到 0.05 也只有 book/cell phone 等误报，定位不可用）；
  - RapidOCR 需要包装上有可读文字，远距离/反光场景失效；
  - 整图 CLIP 只能给一个结论，而一张货架照片里有多个物品。

做法：把照片切成多尺度网格（3x2 + 4x3，带重叠），对每个网格区域批量跑
CLIP 零样本分类；用一组"背景/容器"负向提示词把抽屉、保鲜膜、纯背景等
非物品区域滤掉；存活区域按同类空间重叠聚类合并，每个区域产出
(bbox, 名称, 类别, 置信度)，由 pipeline 生成可确认的物品草稿。

仅在 YOLO 完全漏检且 OCR 没有日期文字聚类时触发（见 pipeline 第 5 步），
单品照片场景不受影响。模型未安装时返回 []，优雅降级。
"""
import threading
from typing import Dict, List, Optional, Tuple

from PIL import Image

from . import clip_cls

# 背景/容器负向提示词：softmax 必选一类，没有负向类时抽屉、保鲜膜、
# 纯背景会被硬分成某个食品（实测白球保鲜膜区域误报"冷冻水饺"0.66）。
_NEG_PROMPTS: List[str] = [
    "an empty refrigerator drawer", "a plain white background",
    "plastic wrap", "an empty transparent plastic container",
    "a metal shelf", "a blurry photo", "an empty plastic bag",
    "a glass jar lid", "a photo of a hand", "an empty plate",
    # 厨房/桌面常见背景：单物体照片（一个苹果、一串香蕉）的网格会把
    # 桌纹、阴影判成"矿泉水/花瓶"等物品（实测苹果照片 8 条垃圾草稿）
    "a wooden table surface", "a kitchen countertop", "a tiled floor",
    "a shadow on the table", "a wooden cutting board", "a stove",
    "a microwave oven", "a window with curtains", "a houseplant",
    "a folded cloth towel", "a stack of plastic bowls",
]

_lock = threading.Lock()
_neg_emb = None            # torch.Tensor [M, D]，已归一化

# 网格配置：(列数, 行数)。3x2 覆盖大件，4x3 补充小件；不叠加 2x2
# （与 3x2 区域高度重复，只会拖慢批量推理）。
_GRIDS: List[Tuple[int, int]] = [(3, 2), (4, 3)]
_OVERLAP = 0.20            # 网格重叠比例：物品压在网格线上时不致被切断

# 区域判定：CLIP 该类得分 >= _THRESH 且领先负向最高分 >= _NEG_MARGIN
_THRESH = 0.45
_NEG_MARGIN = 0.05
_MAX_ITEMS = 8             # 一张照片最多产出的物品区域数

# 区域去重：同类空间重叠说明是同一物品的不同网格切法，合并取并集；
# 异类高度重叠说明网格横跨两个物品，保留高分那个。
# 注意合并阈值取小值：仅合并"同一格子的多种切法"（IoU>0.5 或包含），
# 相邻但不同的包装区域保持独立，用户会得到多个可确认草稿
# （实测冰箱照片 6 个芝士包装区 -> 3 个区域卡片，逐个改名即可）。
_MERGE_IOU = 0.45          # 同类合并阈值
_DROP_IOU = 0.35           # 异类去重阈值

# 多物品切分（split_regions）：固定 4x3 互不重叠网格，一格最多一件物品。
# 900x600 照片每格 225x200，正好是"一包零食/一盒黄油/一块芝士"的尺度。
_SPLIT_COLS, _SPLIT_ROWS = 4, 3
_SPLIT_SCORE = 0.50        # 单元判类得分下限
_SPLIT_NEG_MARGIN = 0.05   # 与负向提示词的最小差距（低于此差值视为背景）
_SPLIT_STRONG = 0.65       # 得分够高时不再看负向差距
_SPLIT_MAX_ITEMS = 6       # 一张照片最多切出的物品区域数

# 统称细化：CLIP 单格判类常落在"水果/蔬菜"这类统称上，而画面里其实是具体
# 品种（苹果/番茄/青椒…）。当某个具体品类的得分接近统称、且自身够高时，
# 用具体品类替代统称，给用户更可用的名称（否则永远是"水果"一个总称）。
_GENERIC_NAMES = {"水果", "蔬菜"}
_SPEC_MIN = 0.35            # 具体品类最低得分（太低不可信，保留统称）
_GENERIC_MARGIN = 0.12      # 具体品类与统称的最大允许分差


def _ensure_neg_emb() -> bool:
    """计算负向提示词嵌入（懒加载，失败返回 False）。"""
    global _neg_emb
    if _neg_emb is not None:
        return True
    with _lock:
        if _neg_emb is not None:
            return True
        if not clip_cls._ensure_loaded():
            return False
        try:
            import torch
            inputs = clip_cls._processor(
                text=[f"a photo of {d}" for d in _NEG_PROMPTS],
                return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                feats = clip_cls._model.text_projection(
                    clip_cls._model.text_model(**inputs).pooler_output)
            _neg_emb = torch.nn.functional.normalize(feats, dim=-1)
            return True
        except Exception:
            return False


def _grid_boxes(w: int, h: int, cols: int, rows: int) -> List[tuple]:
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * w / cols)
            x2 = int((c + 1) * w / cols)
            y1 = int(r * h / rows)
            y2 = int((r + 1) * h / rows)
            px = int((x2 - x1) * _OVERLAP)
            py = int((y2 - y1) * _OVERLAP)
            boxes.append((max(0, x1 - px), max(0, y1 - py),
                          min(w, x2 + px), min(h, y2 + py)))
    return boxes


def _score_crops(crops: List[Image.Image]
                 ) -> Optional[List[Tuple[float, float, int]]]:
    """批量给区域裁剪打分（区域扫描与多物品切分共用一次推理）。

    返回 [(物品得分, 负向最高分, 物品下标)]；模型不可用时返回 None。
    物品得分 = 该物品名下多个提示词的最高分（与 clip_cls.classify 一致），
    负向最高分来自背景/保鲜膜/桌面等负向提示词，用来判"这块其实不是物品"。
    """
    if not crops or not _ensure_neg_emb():
        return None
    try:
        import torch
        inputs = clip_cls._processor(images=[c.convert("RGB") for c in crops],
                                     return_tensors="pt")
        with torch.no_grad():
            feats = clip_cls._model.visual_projection(
                clip_cls._model.vision_model(**inputs).pooler_output)
        embs = torch.nn.functional.normalize(feats, dim=-1)
        pos = torch.softmax((embs @ clip_cls._text_emb.T) * 100.0, dim=-1)
        neg = torch.softmax((embs @ _neg_emb.T) * 100.0, dim=-1).max(dim=-1).values
    except Exception:
        return None
    p2i = clip_cls._prompt_to_item()
    out: List[Tuple[float, float, int, float, int]] = []
    for i in range(len(crops)):
        best: Dict[int, float] = {}
        for idx, item in enumerate(p2i):
            p = float(pos[i][idx])
            if item not in best or p > best[item]:
                best[item] = p
        item_idx, score = max(best.items(), key=lambda kv: kv[1])
        # 同时记录"最佳具体品类"（排除水果/蔬菜统称），用于把统称细化成品种
        spec_idx, spec_score = item_idx, score
        for item, p in best.items():
            if item not in _GENERIC_NAMES and p > spec_score:
                spec_idx, spec_score = item, p
        if item_idx in _GENERIC_NAMES and spec_idx != item_idx:
            out.append((float(score), float(neg[i]), item_idx,
                        float(spec_score), spec_idx))
        else:
            out.append((float(score), float(neg[i]), item_idx,
                        float(score), item_idx))
    return out


def _resolve_name(item_idx: int, spec_idx: int, spec_score: float,
                  score: float) -> int:
    """统称 -> 具体品类：当具体品类得分接近统称且自身够高时替换为具体品类。"""
    name = clip_cls.PROMPT_ITEMS[item_idx][1]
    if (name in _GENERIC_NAMES and spec_idx != item_idx
            and spec_score >= _SPEC_MIN
            and spec_score >= score - _GENERIC_MARGIN):
        return spec_idx
    return item_idx


def _iou(a: tuple, b: tuple) -> float:
    ox = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ox * oy
    if inter <= 0:
        return 0.0
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (sa + sb - inter)


def _union(a: tuple, b: tuple) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


def _contained_ratio(small: tuple, big: tuple) -> float:
    """small 被 big 包含的面积比例（0-1）。用于"小框整体落在大框内"的合并：
    大框是多格合并的并集时 IoU 可能低于阈值，但空间上明显是同一区域。"""
    ox = max(0, min(small[2], big[2]) - max(small[0], big[0]))
    oy = max(0, min(small[3], big[3]) - max(small[1], big[1]))
    inter = ox * oy
    sa = (small[2] - small[0]) * (small[3] - small[1])
    return inter / sa if sa > 0 else 0.0


def scan_regions(pil: Image.Image) -> List[Dict]:
    """
    多尺度网格区域 CLIP 扫描。

    返回 [{bbox:[x1,y1,x2,y2], name, category, score,
           default_shelf_life_days}, ...]，按置信度降序。
    模型不可用或无达标区域时返回 []。
    """
    w, h = pil.size
    if w < 120 or h < 120:               # 图太小，网格无意义
        return []
    boxes: List[tuple] = []
    for cols, rows in _GRIDS:
        boxes.extend(_grid_boxes(w, h, cols, rows))

    scored = _score_crops([pil.crop(b) for b in boxes])
    if not scored:
        return []

    cands: List[Tuple[float, tuple, int]] = []
    for item, box in zip(scored, boxes):
        score, neg, item_idx, spec_score, spec_idx = item
        if score < _THRESH or score - neg < _NEG_MARGIN:
            continue                         # 低置信或负向（背景/容器）胜出
        final_idx = _resolve_name(item_idx, spec_idx, spec_score, score)
        final_score = (spec_score
                       if final_idx == spec_idx
                       and clip_cls.PROMPT_ITEMS[item_idx][1] in _GENERIC_NAMES
                       else score)
        cands.append((final_score, box, final_idx))

    # 分数降序 + 同类合并 / 异类去重
    cands.sort(key=lambda c: -c[0])
    picked: List[Tuple[float, tuple, int]] = []
    for score, box, item_idx in cands:
        merged = False
        for k, (pscore, pbox, pidx) in enumerate(picked):
            same = item_idx == pidx
            if same and (_iou(box, pbox) > _MERGE_IOU
                         or _contained_ratio(box, pbox) > 0.6):
                picked[k] = (max(pscore, score), _union(box, pbox), pidx)
                merged = True
                break
            if not same and _iou(box, pbox) > _DROP_IOU:
                merged = True                 # 横跨两个物品的网格，保留高分
                break
        if not merged:
            picked.append((score, box, item_idx))
        if len(picked) >= _MAX_ITEMS:
            break

    out: List[Dict] = []
    for score, box, item_idx in picked:
        _, name, cname, days = clip_cls.PROMPT_ITEMS[item_idx]
        out.append({
            "bbox": [int(v) for v in box],
            "name": name,
            "category": cname,
            "score": round(float(score), 3),
            "default_shelf_life_days": days,
        })
    out.sort(key=lambda r: -r["score"])
    return out


def _pick_split_cells(scored: List[Tuple[float, float, int, float, int]],
                      cells: List[tuple],
                      max_items: int = _SPLIT_MAX_ITEMS
                      ) -> List[Tuple[float, tuple, int]]:
    """从打分结果里挑出存活单元（纯逻辑，便于单测）。

    规则：单元判类得分 >= _SPLIT_SCORE；负向（背景/保鲜膜/桌面）得分
    压过物品得分时丢弃，但物品得分本身 >= _SPLIT_STRONG 时保留。
    若单元被 CLIP 判成"水果/蔬菜"统称，且某具体品类得分接近，则替换为
    具体品类（见 _resolve_name）。
    结果按阅读顺序（先行后列）返回，最多 max_items 个。
    """
    picked: List[Tuple[float, tuple, int]] = []
    for item, box in zip(scored, cells):
        score, neg, item_idx, spec_score, spec_idx = item
        if score < _SPLIT_SCORE:
            continue
        if score - neg < _SPLIT_NEG_MARGIN and score < _SPLIT_STRONG:
            continue
        final_idx = _resolve_name(item_idx, spec_idx, spec_score, score)
        final_score = (spec_score
                       if clip_cls.PROMPT_ITEMS[item_idx][1] in _GENERIC_NAMES
                       else score)
        picked.append((final_score, box, final_idx))
    picked.sort(key=lambda p: -p[0])          # 先按分数裁掉弱单元
    picked = picked[:max_items]
    picked.sort(key=lambda p: (p[1][1], p[1][0]))
    return picked


def split_regions(pil: Image.Image,
                  max_items: int = _SPLIT_MAX_ITEMS) -> List[Dict]:
    """把一张照片切成互不重叠的多个物品区域（多物品场景）。

    与 scan_regions 的区别：这里用的是固定 4x3 网格、**互不重叠**的单元
    做提案，每个单元单独判类，存活单元本身就代表一个物品区域，不再做
    同类并集合并。scan_regions 的多尺度重叠网格 + 并集合并适合"一件大
    物品被切成多块"的情形，但一张冰箱照片里摆着 5 件东西时，并集会把
    相邻物品连成覆盖大半张图的大框，最后只剩 1-2 个可用区域
    （实测 900x600 冰箱照片：YOLO 0 框、OCR 只读到托盘英文标签，
    整图 CLIP 0.54 判成"芝士/奶酪"，用户只得到 1 个识别区域）。

    只在"整图 CLIP 不自信"时由 pipeline 调用（见 config.MULTI_SCENE_CONF），
    单件物品照片仍走整图结论，不受影响。

    返回 [{bbox, name, category, score, default_shelf_life_days}]，
    按阅读顺序（先行后列）排列；模型不可用或无达标单元时返回 []。
    """
    w, h = pil.size
    if w < 160 or h < 160:               # 太小切了也没意义
        return []
    cells: List[tuple] = []
    for r in range(_SPLIT_ROWS):
        for c in range(_SPLIT_COLS):
            cells.append((int(c * w / _SPLIT_COLS), int(r * h / _SPLIT_ROWS),
                          int((c + 1) * w / _SPLIT_COLS),
                          int((r + 1) * h / _SPLIT_ROWS)))
    scored = _score_crops([pil.crop(b) for b in cells])
    if not scored:
        return []

    picked = _pick_split_cells(scored, cells, max_items)
    if not picked:
        return []

    out: List[Dict] = []
    for score, box, item_idx in picked:
        _, name, cname, days = clip_cls.PROMPT_ITEMS[item_idx]
        out.append({
            "bbox": [int(v) for v in box],
            "name": name,
            "category": cname,
            "score": round(float(score), 3),
            "default_shelf_life_days": days,
        })
    return out


def available() -> bool:
    """CLIP 模型是否可用（区域通道的开关）。"""
    return clip_cls.available()
