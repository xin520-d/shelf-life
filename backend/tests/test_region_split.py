"""多物品切分单测（纯逻辑，无需模型/数据库）。

背景：一张冰箱照片里摆着 5 件东西，YOLO 0 框、OCR 只读到托盘英文标签，
整图 CLIP 只有 54%，旧逻辑只产出 1 条草稿。新增的切分通道按 4x3 互不
重叠网格逐格判类，一件物品一条草稿；本文件锁住"挑格"与"不能被合并回
一条"这两条关键规则。
"""
from app.services.pipeline import _region_pos_label
from app.services.pipeline import _consolidate_drafts
from app.services import region_cls
from app.schemas import DetectionDraft

CELLS = [(0, 0, 225, 200), (225, 0, 450, 200), (450, 0, 675, 200), (675, 0, 900, 200),
         (0, 200, 225, 400), (225, 200, 450, 400), (450, 200, 675, 400),
         (675, 200, 900, 400),
         (0, 400, 225, 600), (225, 400, 450, 600), (450, 400, 675, 600),
         (675, 400, 900, 600)]


def _scored(**slots):
    """slots: 网格下标 -> (score, neg, item_idx) 或 5 元组 (score, neg,
    item_idx, spec_score, spec_idx)，其余格补低分背景。

    具体品类(非统称)的 spec 取自身；统称格若要细化需显式给出具体品类。
    """
    out = [(0.10, 0.80, 0, 0.10, 0) for _ in CELLS]
    for key, val in slots.items():
        if len(val) == 3:
            score, neg, item_idx = val
            out[int(key[1:])] = (score, neg, item_idx, score, item_idx)
        else:
            out[int(key[1:])] = tuple(val)
    return out


def _idx(name: str) -> int:
    for i, (_, n, _, _) in enumerate(region_cls.clip_cls.PROMPT_ITEMS):
        if n == name:
            return i
    raise KeyError(name)


def test_pick_keeps_multiple_items():
    # 冰箱照片实测分数：4 格达标（分别是第1行第3列、第2行第2列、
    # 第3行第1列、第3行第3列）
    scored = _scored(c2=(0.801, 0.384, 4), c5=(0.865, 0.272, 4),
                     c8=(0.976, 0.737, 4), c10=(0.794, 0.947, 4))
    picked = region_cls._pick_split_cells(scored, CELLS)
    assert len(picked) == 4
    # 阅读顺序：先行后列
    assert picked[0][1] == CELLS[2]
    assert picked[-1][1] == CELLS[10]


def test_pick_rejects_low_score_and_background():
    scored = _scored(c0=(0.30, 0.20, 4),      # 分数不够
                     c1=(0.55, 0.75, 4))      # 负向（保鲜膜/抽屉）赢，且不够强
    assert region_cls._pick_split_cells(scored, CELLS) == []


def test_strong_score_survives_negative_prompt():
    # 托盘盖着保鲜膜：负向 0.95，但物品 0.98 足够强，必须保留
    scored = _scored(c8=(0.98, 0.95, 4))
    picked = region_cls._pick_split_cells(scored, CELLS)
    assert [p[1] for p in picked] == [CELLS[8]]


def test_pick_caps_at_max_items():
    scored = _scored(**{f"c{i}": (0.9, 0.1, 4) for i in range(len(CELLS))})
    picked = region_cls._pick_split_cells(scored, CELLS, max_items=3)
    assert len(picked) == 3


def test_generic_refined_to_specific():
    # 某格被 CLIP 判成"水果"0.62，但柠檬 0.55（接近且在阈值内）→ 应为柠檬
    fruit_i = _idx("水果"); lemon_i = _idx("柠檬")
    scored = _scored(c0=(0.62, 0.10, fruit_i, 0.55, lemon_i))
    picked = region_cls._pick_split_cells(scored, CELLS)
    assert picked and picked[0][2] == lemon_i


def test_generic_kept_when_specific_too_low():
    # 某格"蔬菜"0.62，但最高具体品类只有 0.30（低于 _SPEC_MIN）→ 保留蔬菜
    veg_i = _idx("蔬菜"); tomato_i = _idx("西红柿/番茄")
    scored = _scored(c1=(0.62, 0.10, veg_i, 0.30, tomato_i))
    picked = region_cls._pick_split_cells(scored, CELLS)
    assert picked and picked[0][2] == veg_i


def test_region_pos_label():
    assert _region_pos_label([0, 400, 225, 600], 900, 600) == "第3行第1列"
    assert _region_pos_label([450, 0, 675, 200], 900, 600) == "第1行第3列"
    assert _region_pos_label([], 900, 600) == ""


def _draft(name, bbox, region=False, conf=0.8):
    return DetectionDraft(name=name, category_id=1, category_name="食品",
                          bbox=bbox, confidence=conf, region_split=region)


def test_adjacent_same_name_regions_are_kept_apart():
    # 冰箱里并排两包同款芝士：两条切分草稿必须各留一条
    a = _draft("芝士/奶酪", [0, 400, 225, 600], region=True, conf=0.9)
    b = _draft("芝士/奶酪", [225, 400, 450, 600], region=True, conf=0.8)
    out = _consolidate_drafts(None, [a, b], [], [], is_screenshot=False)
    assert len(out) == 2


def test_duplicate_same_spot_regions_merge():
    a = _draft("芝士/奶酪", [0, 400, 225, 600], region=True, conf=0.9)
    b = _draft("芝士/奶酪", [2, 402, 227, 601], region=True, conf=0.8)
    out = _consolidate_drafts(None, [a, b], [], [], is_screenshot=False)
    assert len(out) == 1


def test_plain_adjacent_drafts_still_merge():
    # 原有行为不能退化：YOLO 把一个苹果拆成两个紧贴的框时要并回一条
    a = _draft("苹果", [0, 0, 100, 100], conf=0.9)
    b = _draft("苹果", [100, 0, 200, 100], conf=0.8)
    out = _consolidate_drafts(None, [a, b], [], [], is_screenshot=False)
    assert len(out) == 1
