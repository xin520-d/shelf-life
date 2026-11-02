"""点阵喷码日期识别（进口商品常见的 PD:/ED:、MFG DATE/EXP DATE 墨点喷码）。

背景：点阵字符的墨点彼此不连通，RapidOCR 的文本检测器（DBNet）需要连通
笔画才能框出文字行，对点阵喷码完全失效——实测某黄油包装顶部的
"PD: 2015 03 06 20H51 / ED: 2016 03 05 A5 P5" 只被读出 "39 33" 垃圾；
某印尼曲奇包装底部的 "MFG DATE 20180326 / EXP DATE 20190326"（字高仅
~12px、两行几乎相贴）更是一行都读不出。本模块绕过检测器，按两条路径：

  路径一（提示区优先，曲奇实测）：
    常规 OCR 在日期区留下的检测框（即便文本读糊，框的位置也是对的），
    在框内用**一行高的窄条**（32px）步进 8px 扫描——条高必须贴合行高，
    120px 的宽条会把多行文字和周边印刷一起塞进识别器，横向拍扁后数字
    全糊（实测整行读出 "2U190326"、"201809326"）。每条窄条：
      a. 整读一次取锚点标签（MFG/PROD -> 生产，EXP/BEST -> 到期）；
      b. 列投影把条内墨迹切成字符块、块间隙 <=3px 的合并成"词"，
         >=5 块的词（典型是 6-8 位数字串）单独裁剪放大专读——数字串
         独占一幅时宽高比贴近识别器内部 48x320 目标尺度，读数远比
         整行准（实测 "20180326"/"20190326" 置信度 0.81-0.87）。
    日期提取走双通道：分隔符兼容（"PD: 2015 03 06"）+ token 级数字
    混淆归一化（U/O->0、S->5、I/l->1、Z->2、B->8、G->6、T->7、A->4，
    修 "2U190326"->2019-03-26）；7 位/9+ 位的数字串是先屏蔽的噪声
    （点阵 OCR 常丢/多认一位，直接解析会产出 2018-09-03 这类假日期）。

  路径二（墨迹带兜底，黄油实测）：
    提示区无果时退回自适应阈值二值化 + 连通域分析找"墨点碎片"，
    按 y 投影聚成文字带，带上用 120px 条 + 行投影分段扫描，
    每行自动裁剪到墨迹边界 + 3 倍放大后直读。

  两条路径的命中按"日期合法性 + 置信度加权"投票，两条日期时按锚点
  标签指派生产/到期（标签缺失才按"上=生产/下=到期"的印刷惯例），
  重构出 "PD: yyyy mm dd" 规范文本交回日期解析引擎处理。

仅在常规 OCR 读不到日期时由流水线启用（兜底通道，不影响常规性能）。
"""
import datetime as dt
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover - cv2 是 rapidocr 的依赖，正常必装
    _HAS_CV2 = False

# ---------------------------------------------------------------- 参数
_FRAG_AREA = (4, 3000)        # 单个墨点碎片的面积范围（原图像素；小点阵字墨点仅 ~2x2px）
_FRAG_SIDE = (2, 100)         # 碎片边长范围（2px 墨点需要放到 2，原为 3）
_BAND_MIN_FRAGS = 8           # 文字带至少包含的碎片数
_BAND_H = (14, 420)           # 文字带高度范围（原 30；小日期行整行高仅 ~18px）
_BAND_TOP_N = 4               # 按密度取前 N 个带扫描（点阵每字符碎成多块，密度高）
_STRIP_H = 120                # 墨迹带路径的滑条高度（原图像素）
_STRIP_STEP = 8               # 滑动步长（原 10：更密，增加"正好框住单行"的机会）
_STRIP_Y_BEFORE = 12          # 条顶相对带顶的外扩
_STRIP_Y_AFTER = 20           # 条底相对带底的外扩
_STRIP_X_PAD = 16             # 条两侧外扩
_STRIP_MAX_ASPECT = 20        # 宽高比上限（超出会被识别器横向挤压丢数字）
_CROP_MARGIN = 6              # 墨迹自动裁剪留边（实测行上下需 ~5px 背景读数才稳）
_UPSCALE = 3                  # 读条放大倍数（原 2：实测 3x 时 ~12px 高点阵字读数最准，
                              #  字符放大到 ~36px ≈ 识别器内部 48px 目标，过大反而糊）
_MIN_SCORE = 0.15             # 读数置信度下限
_MAX_READS = 260              # 单张图最多送识别器的条数（窄条+run 专读次数翻倍，上调护栏）
_YEAR_RANGE = (1990, 2040)    # 合理年份范围（用于过滤噪声读数）
_SEG_GAP = 4                  # 行投影分段合并间隙（原图像素）：行内墨点间隙 <4，行间 >5
_MAX_SEGS = 6                 # 单条最多分段数（一条喷码带通常 2-3 行：标签+MFG+EXP，
                              #  放宽到 6 是给噪声行留余量，读数上限由 _MAX_READS 控制）
_SEG_MAX_H = 16               # 行分段时忽略的组件高度上限：条码竖条/大色块/印刷弧线等
                              #  高组件会从带顶贯穿到带底，把每行都连成一整段
_SEG_MAX_AREA = 600           # 行分段时忽略的组件面积上限
_SEG_MIN_INK = 12             # 分段至少包含的墨迹像素（再小的当噪声丢弃）

# 提示区窄条扫描参数：提示框来自常规 OCR 的检测框，通常已框住日期行主体。
_MAX_HINTS = 6                # 最多跟随的提示框数
_HINT_X_PAD = 40              # 提示框两侧外扩（点阵行常比 DBNet 框更宽）
_HINT_Y_UP = 20               # 提示框上外扩（框住行上方留边）
_HINT_Y_DOWN = 12             # 提示框下外扩
_LINE_STRIP_H = 32            # 窄条高度：贴合一行点阵字高（曲奇 ~24px、黄油 ~30px），
                              #  条内只放一行文字，识别器才不会把多行拍扁
_LINE_STRIP_STEP = 8          # 窄条步长（行高 ~24px，步进 8 保证至少一条正好框住整行）
_LINE_STRIP_MAX_W = 640       # 窄条宽度上限（再宽横向压缩比飙升，数字必糊）
_RUN_MIN_BLOCKS = 5           # 词 run 至少 5 个字符块才专读（日期数字串 6-8 块；
                              #  MFG/DATE 等字母词 3-4 块，不浪费读数）
_RUN_GAP = 3                  # 字符块间隙 <=3px 视为同一个词（行内字距 <3，词间 >6）
_RUN_CROP_MARGIN = 3          # run 裁剪留边

# 日期提取：4 位年 + 月 + 日（容忍空格/点/斜杠/年月日分隔，点阵 OCR 常丢分隔符）
_DATE_RE = re.compile(r"(\d{4})\s*[-/.年]?\s*(\d{1,2})\s*[-/.月]?\s*(\d{1,2})")
# 紧凑 8 位日期
_COMPACT_RE = re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
# 行首标签提示：PD/ED 的误识形态（P0/Pn/En/E0...）
_LABEL_RE = re.compile(r"^\W{0,3}([A-Za-z])")
# 锚点词标签：MFG/PROD/MANUF -> 生产；EXP/BEST BEFORE/USE BY -> 到期。
# 点阵 OCR 常把锚点读糊（"MFG DATE"->"MFG DAIE"），按子串匹配容错。
_RE_LABEL_PROD = re.compile(r"MFG|PROD|MANUF|生产|\bPD\b", re.I)
_RE_LABEL_EXP = re.compile(r"EXP|BEST|USE\s*BY|保质|\bED\b", re.I)
# token 级数字混淆归一化：点阵小字的墨点残缺让识别器把 0 认成 U/O、
# 5 认成 S、1 认成 I/l、2 认成 Z、8 认成 B、6 认成 G、7 认成 T、4 认成 A。
# 只对"含数字的 token"生效，MFG/EXP/DATE 等纯字母标签不受影响。
_DIGIT_CONFUSION = str.maketrans({
    "U": "0", "O": "0", "o": "0", "Q": "0",
    "S": "5", "s": "5", "I": "1", "l": "1", "i": "1", "|": "1", "!": "1",
    "Z": "2", "z": "2", "B": "8", "G": "6", "b": "6", "T": "7", "A": "4",
    "g": "9", "q": "9",
})
# 噪声数字串：点阵 OCR 常丢一位（8 位读成 7 位）或多认一位（读成 9 位），
# 这种串直接按 YYYYMMDD 解析会产出"2018-09-03"这类以假乱真的日期，先屏蔽。
_RE_NOISY_DIGITS = re.compile(r"(?<!\d)\d{7}(?!\d)|(?<!\d)\d{9,}(?!\d)")
_RE_TOKEN = re.compile(r"[0-9A-Za-z|!]+")


def _fragments(gray: "np.ndarray") -> List[tuple]:
    """二值化 + 连通域分析，返回墨点碎片 [(x, y, w, h), ...]。

    必须用自适应阈值：全局 Otsu 在明暗不均的包装图上会把背景误判为墨点，
    导致点阵行的碎片互相粘连（实测行1碎片从 81 掉到 13，整行消失）。
    """
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, 10)
    ink = (bw == 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if _FRAG_AREA[0] <= area <= _FRAG_AREA[1] and \
           _FRAG_SIDE[0] <= w <= _FRAG_SIDE[1] and _FRAG_SIDE[0] <= h <= _FRAG_SIDE[1]:
            out.append((x, y, w, h))
    return out


def _bands(frags: List[tuple], height: int) -> List[dict]:
    """把墨点碎片聚成文字带。

    用"每行被多少碎片覆盖"的投影而非碎片中心聚类：点阵字符行之间有
    空隙，中心单链聚类会把整张图的碎片链成一个超高带（实测被高度过滤
    后整行消失）；投影平滑后能自然桥接行内空隙、跨行间断开。
    """
    if len(frags) < _BAND_MIN_FRAGS:
        return []
    cover = np.zeros(height + 2, dtype=np.int32)
    for (x, y, w, fh) in frags:
        cover[y:y + fh + 1] += 1
    # 平滑：桥接点阵字符行间的空隙（窗口 ~ 半个字符高）
    k = 15
    sm = np.convolve(cover, np.ones(k) / k, mode="same")
    active = sm >= 2.0
    runs: List[List[int]] = []
    y = 0
    while y < len(active):
        if active[y]:
            y2 = y
            while y2 < len(active) and active[y2]:
                y2 += 1
            runs.append([y, y2 - 1])
            y = y2
        else:
            y += 1
    # 合并间隔很小的段（同一文字带的抖动）
    merged: List[List[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 12:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    out = []
    for (y1, y2) in merged:
        if not (_BAND_H[0] <= y2 - y1 + 1 <= _BAND_H[1]):
            continue
        fs = [f for f in frags if f[1] <= y2 and f[1] + f[3] >= y1]
        if len(fs) < _BAND_MIN_FRAGS:
            continue
        out.append({"y1": y1, "y2": y2,
                    "x1": min(f[0] for f in fs),
                    "x2": max(f[0] + f[2] for f in fs),
                    "n": len(fs)})
    # 密度高的带更可能是点阵喷码（点阵每个字符碎成多块）
    out.sort(key=lambda b: -(b["n"] / max(1.0, b["y2"] - b["y1"])))
    return out[:_BAND_TOP_N]


def _clean_ink(ink: "np.ndarray") -> "np.ndarray":
    """去掉跨行的巨大组件（条码竖条、大色块、印刷弧线），只留字符/墨点级组件。

    行投影分段依赖"行与行之间有空白行"：条码竖条从带顶贯穿到带底，
    会把所有文本行连成一整段，送识别器后因宽高比失衡完全读不出
    （实测曲奇包装底部：条码 + 黑带 + 红色印刷弧线导致分段失效）。
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8),
                                                        connectivity=8)
    keep = [i for i in range(1, n)
            if stats[i][cv2.CC_STAT_HEIGHT] <= _SEG_MAX_H
            and stats[i][cv2.CC_STAT_AREA] <= _SEG_MAX_AREA]
    if not keep:
        return ink
    return np.isin(lab, keep)


def _read_crop(rec, gray: "np.ndarray") -> Tuple[str, float]:
    """灰度裁剪 -> 3 倍放大 -> Otsu 二值化 -> 直接送识别器。

    放大放在二值化之前：墨点先放大再阈值，点阵笔画之间的空隙被
     Lanczos 插值桥接一部分，比"先阈值后放大"的硬边更接近印刷体。
    """
    if gray is None or gray.size == 0:
        return "", 0.0
    up = cv2.resize(gray, None, fx=_UPSCALE, fy=_UPSCALE,
                    interpolation=cv2.INTER_LANCZOS4)
    _, bw = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res, _ = rec([cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)])
    if res and res[0][0]:
        return str(res[0][0]), float(res[0][1])
    return "", 0.0


def _word_runs(clean: "np.ndarray") -> List[Tuple[int, int, int]]:
    """列投影切字符块，块间隙 <= _RUN_GAP 的合并为同一个词 run。

    返回 [(x1, x2, 块数), ...]。点阵日期的数字串（6-8 个连续窄块）
    会形成独立 run，与 MFG/DATE 等字母词自然分开——数字串单独送
    识别器时宽高比贴近 48x320 目标尺度，读数远比整行准。
    """
    col = clean.sum(axis=0)
    blocks: List[Tuple[int, int]] = []
    inb = False
    start = 0
    for i, v in enumerate(col):
        if v > 0 and not inb:
            start = i
            inb = True
        elif v == 0 and inb:
            blocks.append((start, i))
            inb = False
    if inb:
        blocks.append((start, len(col)))
    runs: List[List[int]] = []
    for (b0, b1) in blocks:
        if runs and b0 - runs[-1][1] <= _RUN_GAP:
            runs[-1][1] = b1
            runs[-1][2] += 1
        else:
            runs.append([b0, b1, 1])
    return [(r[0], r[1], r[2]) for r in runs]


def _read_line_strip(rec, arr: "np.ndarray", y: int, x1: int, x2: int
                     ) -> Tuple[int, List[dict]]:
    """读一条窄横条（高约一行点阵字）：整读取标签 + 词 run 专读取日期。

    返回 (消耗读数次数, hits)；hits = [{date, score, label, y, text, bbox}]。
    run 专读出了日期就不再用整读的兜底日期（整读的数字串常多位/漏位，
    而单独的 8 位数字串run 读数最稳——实测曲奇两行日期 0.81-0.87）。
    """
    h = min(_LINE_STRIP_H, arr.shape[0] - y)
    if h < 8:
        return 0, []
    strip = arr[y:y + h, x1:x2]
    if strip.size == 0:
        return 0, []
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = bw == 0
    if ink.mean() < 0.004:          # 几乎全是背景：没有墨迹，不白送识别器
        return 0, []
    clean = _clean_ink(ink)
    if not clean.any():
        return 0, []

    reads = 1
    # 1) 整读：锚点标签（MFG/EXP/PD/ED）+ 兜底日期
    text, score = _read_crop(rec, gray)
    label = _label_hint(text) if text else None

    # 2) 词 run 专读：>=5 块的词（典型 6-8 位数字串）
    run_hits: List[dict] = []
    for (c0, c1, nb) in _word_runs(clean):
        if nb < _RUN_MIN_BLOCKS:
            continue
        sub = clean[:, c0:c1]
        rows = np.where(sub.any(axis=1))[0]
        if len(rows) == 0:
            continue
        rr0 = max(0, int(rows[0]) - _RUN_CROP_MARGIN)
        rr1 = min(gray.shape[0], int(rows[-1]) + _RUN_CROP_MARGIN + 1)
        cc0 = max(0, c0 - _RUN_CROP_MARGIN)
        cc1 = min(gray.shape[1], c1 + _RUN_CROP_MARGIN)
        crop = gray[rr0:rr1, cc0:cc1]
        if crop.shape[0] < 6 or crop.shape[1] < 6:
            continue
        rtext, rscore = _read_crop(rec, crop)
        reads += 1
        if not rtext or rscore < _MIN_SCORE:
            continue
        for d in _dates_from_token(rtext):
            run_hits.append({"date": d, "score": rscore, "label": label,
                             "y": y, "text": rtext,
                             "bbox": (x1 + cc0, y + rr0, x1 + cc1, y + rr1)})

    hits = list(run_hits)
    if not hits and text and score >= _MIN_SCORE:
        # 3) run 没出日期时才用整读兜底（黄油 "PD: 2015 03 06" 靠这条：
        #    日期是空格分隔的，run 按词切开后每段都不足 5 块）
        for d in _strip_dates(text):
            hits.append({"date": d, "score": score, "label": label, "y": y,
                         "text": text, "bbox": (x1, y, x2, y + h)})
    return reads, hits


def _read_segments(rec, arr: "np.ndarray", y1: int, y2: int, x1: int, x2: int
                   ) -> List[Tuple[str, float, tuple]]:
    """读一条宽条（墨迹带路径）：Otsu 二值化 + 清跨行组件后按行投影切成
    若干文字行，逐行裁剪放大识别。

    返回 [(文本, 置信度, bbox), ...]；空条返回 []。

    为什么要分段：一条喷码带常含多行（预印标签行 + MFG 行 + EXP 行），
    整条送识别器会因宽高比失衡被横向拍扁，实测直接返回空；按行投影切开后
    每行单独自动裁剪到墨迹边界 + 留边，读数才稳（实测 "EXP DATE 20190326"
    置信度 0.91）。
    """
    strip = arr[y1:y2, x1:x2]
    if strip.size == 0:
        return []
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = bw == 0
    if ink.mean() < 0.004:          # 几乎全是背景：没有墨迹，不白送识别器
        return []
    clean = _clean_ink(ink)
    rows = np.where(clean.any(axis=1))[0]
    if len(rows) == 0:
        return []
    # 行投影分段：行内墨点间隙小（<_SEG_GAP），行间间隙大（>_SEG_GAP）
    segs: List[List[int]] = []
    start = prev = int(rows[0])
    for r in rows[1:]:
        r = int(r)
        if r - prev <= _SEG_GAP:
            prev = r
        else:
            segs.append([start, prev])
            start = prev = r
    segs.append([start, prev])

    out: List[Tuple[str, float, tuple]] = []
    for (r0, r1) in segs[:_MAX_SEGS]:
        band = clean[r0:r1 + 1]
        if int(band.sum()) < _SEG_MIN_INK:
            continue
        cols = np.where(band.any(axis=0))[0]
        if len(cols) == 0:
            continue
        c0 = max(0, int(cols[0]) - _CROP_MARGIN)
        c1 = min(gray.shape[1], int(cols[-1]) + _CROP_MARGIN + 1)
        rr0 = max(0, r0 - _CROP_MARGIN)
        rr1 = min(gray.shape[0], r1 + _CROP_MARGIN + 1)
        tight = gray[rr0:rr1, c0:c1]
        if tight.shape[0] < 6 or tight.shape[1] < 6:
            continue
        text, score = _read_crop(rec, tight)
        if not text:
            continue
        bbox = (int(x1 + c0), int(y1 + rr0), int(x1 + c1), int(y1 + rr1))
        out.append((text, score, bbox))
    return out


def _date_candidates(text: str) -> List[tuple]:
    """从读出中提取合法日期候选 [(y, m, d), ...]（分隔符兼容通道）。"""
    out: List[tuple] = []
    for m in _DATE_RE.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _YEAR_RANGE[0] <= y <= _YEAR_RANGE[1] and 1 <= mo <= 12 and 1 <= d <= 31:
            out.append((y, mo, d))
    for m in _COMPACT_RE.finditer(text):
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def _dates_from_token(text: str) -> List[tuple]:
    """token 级日期提取：含数字的 token 做混淆归一化后按 8/6 位精确解析。

    修点阵小字的系统性误读："2U190326"->2019-03-26、"190326"->2019-03-26。
    只认 8 位（YYYYMMDD）和 6 位（YYMMDD）：7 位是丢位、9 位是多认位，
    这两种噪声串交给调用方屏蔽，不在此处猜。
    """
    out: List[tuple] = []
    for tok in _RE_TOKEN.findall(text):
        if not any(c.isdigit() for c in tok):
            continue
        dg = re.sub(r"\D", "", tok.translate(_DIGIT_CONFUSION))
        if len(dg) == 8:
            y, m, d = int(dg[:4]), int(dg[4:6]), int(dg[6:])
        elif len(dg) == 6:
            y, m, d = 2000 + int(dg[:2]), int(dg[2:4]), int(dg[4:])
        else:
            continue
        if _YEAR_RANGE[0] <= y <= _YEAR_RANGE[1] and 1 <= m <= 12 and 1 <= d <= 31:
            out.append((y, m, d))
    return out


def _strip_dates(text: str) -> List[tuple]:
    """整行/整条读出的日期提取：先屏蔽 7/9+ 位噪声数字串，再走
    分隔符兼容（黄油 "PD: 2015 03 06"）+ 混淆归一化（"2U190326"）双通道。"""
    if not text:
        return []
    t = _RE_NOISY_DIGITS.sub(" ", text)
    ds = _date_candidates(t)
    for d in _dates_from_token(t):
        if d not in ds:
            ds.append(d)
    return ds


def _label_hint(text: str) -> Optional[str]:
    """从喷码读出文本判断该日期是生产（PD）还是到期（ED）。

    优先按锚点词：MFG/PROD/MANUF -> PD，EXP/BEST BEFORE/USE BY -> ED
    （点阵 OCR 常把锚点读糊，如 "MFG DATE"->"MFG DAIE"，按子串匹配容错）；
    锚点完全读不出时退化为行首字母（P->PD，E->ED）。
    """
    if _RE_LABEL_PROD.search(text):
        return "PD"
    if _RE_LABEL_EXP.search(text):
        return "ED"
    m = _LABEL_RE.match(text.strip())
    if not m:
        return None
    c = m.group(1).upper()
    if c == "P":
        return "PD"
    if c == "E":
        return "ED"
    return None


def _major_label(labels: List[str]) -> Optional[str]:
    """取出现次数最多的标签（PD/ED），没有则 None。"""
    if not labels:
        return None
    return max(set(labels), key=labels.count)


def _distinct_dates(hits: List[dict]) -> int:
    """已获得的合法日期种数（只数置信度达标的命中）。"""
    return len({h["date"] for h in hits if h["score"] >= _MIN_SCORE})


def _scan_hint_region(rec, arr: "np.ndarray", h: int, w: int, box: tuple,
                      hits: List[dict], budget: int) -> int:
    """提示框区域内窄条扫描，返回消耗的读数次数。"""
    hx1, hy1, hx2, hy2 = box
    x1 = max(0, int(hx1) - _HINT_X_PAD)
    x2 = min(w, int(hx2) + _HINT_X_PAD)
    if x2 - x1 > _LINE_STRIP_MAX_W:
        x2 = x1 + _LINE_STRIP_MAX_W
    y = max(0, int(hy1) - _HINT_Y_UP)
    stop = min(h - _LINE_STRIP_H, int(hy2) + _HINT_Y_DOWN)
    reads = 0
    while y <= stop and reads < budget:
        n, strip_hits = _read_line_strip(rec, arr, y, x1, x2)
        reads += n
        hits.extend(strip_hits)
        y += _LINE_STRIP_STEP
    return reads


def _scan_strips(rec, arr: "np.ndarray", h: int, w: int,
                 y_start: int, y_end: int, x1: int, x2: int,
                 hits: List[dict], budget: int, step: int) -> int:
    """在 [y_start, y_end] 内以 _STRIP_H 高的滑条扫描（墨迹带路径），
    把日期命中追加进 hits。返回消耗的读数次数（受 budget 限制）。"""
    reads = 0
    sh = min(_STRIP_H, h)
    y = max(0, y_start)
    stop = min(h - sh, y_end)
    while y <= stop and reads < budget:
        for (text, score, bbox) in _read_segments(rec, arr, y, y + sh, x1, x2):
            reads += 1
            if score >= _MIN_SCORE:
                for cand in _strip_dates(text):
                    hits.append({"date": cand, "score": score, "y": y,
                                 "label": _label_hint(text), "text": text,
                                 "bbox": bbox})
        y += step
    return reads


def read_dotmatrix_dates(pil_image, hint_boxes: Optional[List[tuple]] = None) -> List[Dict]:
    """
    点阵喷码日期兜底识别。

    hint_boxes：常规 OCR 在疑似日期区留下的检测框 [(x1,y1,x2,y2), ...]
    （由流水线从含日期关键词/日期样数字的 OCR 行提取，已按可能性排序）。
    DBNet 读不出点阵文字，但框的位置通常是对的，优先用一行高窄条扫这些
    区域可大幅提速提精度；提示区没凑齐两条日期时退回全图墨迹带扫描。

    返回重构出的规范文本行 [{box, text, score}, ...]（如 "PD: 2015 03 06"），
    直接并入全文 OCR 结果即可被日期引擎解析；未识别到时返回空列表。
    """
    if not _HAS_CV2:
        return []
    from .ocr import get_engine
    try:
        rec = get_engine().text_recognizer
    except Exception:
        return []

    arr = np.asarray(pil_image.convert("RGB"))
    h, w = arr.shape[:2]
    if h < 120 or w < 120:
        return []

    # 1. 提示区窄条扫描（常规 OCR 框住的疑似日期区）
    hits: List[dict] = []
    reads = 0
    for box in (hint_boxes or [])[:_MAX_HINTS]:
        if reads >= _MAX_READS or _distinct_dates(hits) >= 2:
            break
        reads += _scan_hint_region(rec, arr, h, w, box, hits,
                                   _MAX_READS - reads)

    # 2. 没凑齐两条日期：退回墨迹带扫描（黄油等提示框框不住喷码行的场景）
    if _distinct_dates(hits) < 2 and reads < _MAX_READS:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        for b in _bands(_fragments(gray), h):
            if reads >= _MAX_READS or _distinct_dates(hits) >= 2:
                break
            x1 = max(0, b["x1"] - _STRIP_X_PAD)
            x2 = min(w, b["x2"] + _STRIP_X_PAD)
            if x2 - x1 > _STRIP_H * _STRIP_MAX_ASPECT:
                x2 = x1 + _STRIP_H * _STRIP_MAX_ASPECT
            reads += _scan_strips(rec, arr, h, w,
                                  b["y1"] - _STRIP_Y_BEFORE,
                                  b["y2"] - _STRIP_H + _STRIP_Y_AFTER,
                                  x1, x2, hits, _MAX_READS - reads,
                                  _STRIP_STEP)

    if not hits:
        return []

    # 3. 按日期聚类，置信度加权
    groups: Dict[tuple, dict] = {}
    for hit in hits:
        g = groups.setdefault(hit["date"], {"score": 0.0, "ys": [], "labels": [],
                                            "bbox": hit["bbox"], "text": hit["text"]})
        g["score"] += hit["score"]
        g["ys"].append(hit["y"])
        if hit["label"]:
            g["labels"].append(hit["label"])
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["score"])

    # 4. 指派生产/到期：两条日期时取分数最高且垂直不重叠的两簇。
    #    两条都有锚点标签（MFG/EXP）时按标签指派——显式锚点优先于
    #    "上=生产下=到期"的印刷惯例；仅一条有标签或都无标签时才按位置，
    #    并用时序校验兜底（生产必须不晚于到期）。
    today = dt.date.today()
    prod = exp = None
    if len(ranked) >= 2:
        best_date, best = ranked[0]
        second = None
        for date, g in ranked[1:]:
            if min(g["ys"]) - max(best["ys"]) > 20 or \
               min(best["ys"]) - max(g["ys"]) > 20:
                second = (date, g)
                break
        if second is None:
            second = ranked[1]
        lb, ls = _major_label(best["labels"]), _major_label(second[1]["labels"])
        if lb and ls and lb != ls:
            prod, exp = ((best_date, second[0]) if lb == "PD"
                         else (second[0], best_date))
        else:
            pair = sorted([best_date, second[0]],
                          key=lambda d: (min(groups[d]["ys"]), d))
            prod, exp = pair[0], pair[1]
            if prod > exp:                  # 时序校验：生产必须早于到期
                prod, exp = exp, prod
    else:
        (date, g), = ranked
        label = _major_label(g["labels"])
        if label == "ED" or dt.date(*date) > today:
            exp = date
        else:
            prod = date

    out: List[Dict] = []
    for kind, date in (("PD", prod), ("ED", exp)):
        if not date:
            continue
        g = groups[date]
        bx1, by1, bx2, by2 = g["bbox"] if g["bbox"] else (0, 0, 0, 0)
        # 4 点框，与常规 OCR 结果格式一致（流水线按 4 点算中心点）
        quad = [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]
        out.append({
            "box": quad,
            "text": f"{kind}: {date[0]:04d} {date[1]:02d} {date[2]:02d}",
            "score": round(min(g["score"], 1.0), 3),
        })
    return out
