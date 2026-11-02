"""
日期文本归一化规则引擎（自主开发模块）
覆盖中文包装常见写法：
  生产日期：2024-01-15 / 2024.01.15 / 2024/1/5 / 2024年1月15日 / 20240115
  保质期：12 个月 / 6个月 / 180天 / 2年
  保质期至：2027.03 / 2027年3月 / 2027-03-31
  英文：BEST BEFORE 20250115 / EXP 2025.06 / MFG 20240115
        MFG DATE 20180326 / EXP DATE 2019 03 26（锚点后带 DATE 词亦支持）
解析优先级（机器建议，用户可改）：
  1. 显式到期日（保质期至 X）          -> expiry = X
  2. 生产日期 + 保质期时长             -> expiry = 生产日期 + N 个月/天
  3. 仅生产日期                       -> expiry = 生产日期 + 类别默认天数（由调用方补）
  4. 仅保质期时长                     -> 建议 expiry = 今天 + N（标注"按今天生产估算"）
  5. 无日期                           -> 需人工填写
"""
import datetime as dt
import re
from dataclasses import dataclass, field
from typing import List, Optional

MIN_REASONABLE_DATE = dt.date(1990, 1, 1)
MAX_REASONABLE_DATE = dt.date(2100, 12, 31)

# ---------------------------------------------------------------- 文本归一化

_FULLWIDTH_MAP = {ord(f): ord(t) for f, t in
                  zip("０１２３４５６７８９－／．：年月日", "0123456789-/.::YMD")}
_FULLWIDTH_MAP = {ord("０"): "0", ord("１"): "1", ord("２"): "2", ord("３"): "3",
                  ord("４"): "4", ord("５"): "5", ord("６"): "6", ord("７"): "7",
                  ord("８"): "8", ord("９"): "9", ord("－"): "-", ord("／"): "/",
                  ord("．"): ".", ord("："): ":", ord("年"): "年", ord("月"): "月", ord("日"): "日"}

_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "Ｏ": "0", "l": "1", "I": "1", "|": "1",
                            "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2",
                            "g": "9", "q": "9"})


def normalize_text(s: str) -> str:
    """全角转半角、空白压缩为单空格。

    注意：必须保留空格——OCR 多尺度识别会把日期拆成独立行，
    去空白会把两个日期数字粘成一串，破坏紧凑日期的边界判断。
    """
    s = s.translate(_FULLWIDTH_MAP)
    return re.sub(r"\s+", " ", s or "").strip()


def _fix_digits(tok: str) -> str:
    """修复 OCR 常见数字误识别（仅对日期 token 生效）。"""
    return tok.translate(_DIGIT_FIX)


# ---------------------------------------------------------------- 基础日期工具

def _valid(d: dt.date) -> bool:
    return MIN_REASONABLE_DATE <= d <= MAX_REASONABLE_DATE


def _mk(year: int, month: int, day: int) -> Optional[dt.date]:
    try:
        d = dt.date(year, month, day)
        return d if _valid(d) else None
    except ValueError:
        return None


def _end_of_month(d: dt.date) -> dt.date:
    nxt = dt.date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
    return nxt - dt.timedelta(days=1)


def add_months(d: dt.date, months: int) -> dt.date:
    """日历月推算：1月31日 + 1个月 = 2月28/29日（钳制到月末）。"""
    total = (d.year * 12 + (d.month - 1)) + months
    y, m = divmod(total, 12)
    m += 1
    last = _end_of_month(dt.date(y, m, 1)).day
    return dt.date(y, m, min(d.day, last))


def add_days(d: dt.date, days: int) -> dt.date:
    return d + dt.timedelta(days=days)


# ---------------------------------------------------------------- 正则模式

_DATE_CORE = r"(\d{4})[-/.年](\d{1,2})(?:[-/.月](\d{1,2})日?)?"
# 日期值：紧凑8位 / 带分隔符 / 空格分隔（点阵喷码拆行形态 "2015 03 06"）。
# 注意“月日”组必须整体可选，否则 "2025.06" 类年月写法匹配不到。
_DATE_VALUE = r"(?:\d{8}|\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2})?日?|20\d{2}\s+(?:0[1-9]|1[0-2])\s+(?:0[1-9]|[12]\d|3[01]))"
# 英文锚点后常跟 "DATE" 一词（"MFG DATE 20180326"），且日期常为紧凑 8 位，
# 旧正则要求日期紧贴关键词且只认带分隔符写法，整行漏读（实测曲奇包装）。
# 关键词与日期之间允许 [:：.] 与 DATE/DAY/DT 词（PRODUCT 必须排在 PROD 前）。
_RE_EN_PROD = re.compile(
    r"(?:MFG|MFD|PRODUCT|PROD|PRODUCTION|MANUFACTURED?|MANUFACTURING)"
    r"\s*[:：.]?\s*(?:DATE|DAY|DT)?\s*[:：.]?\s*(" + _DATE_VALUE + r")", re.I)
_RE_EN_EXP = re.compile(
    r"(?:EXPIRY|EXPIRES?|EXP|BEST\s*BEFORE|BEST\s*BY|USE\s*BY)"
    r"\s*[:：.]?\s*(?:DATE|DAY|DT)?\s*[:：.]?\s*(" + _DATE_VALUE + r")", re.I)
# 分隔符字符类含中文逗号"，"：标签常印"保质期至，YYYY年MM月DD日"（实测
# 维生素C 标签），无逗号时"保质期至2024..."也覆盖。
_RE_EXPIRY_TO = re.compile(r"(?:保质期|保质日期|保质期至|保质至|保质到|最佳食用期|最佳保鲜期|有效期|有效期至|有效期到|保质期限)[至到:：,，]?\s*(\d{8}|\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?)")
# "产日期"不强制首字：OCR 常把"生"误读成"日/土/王/先"（实测"生产日期"读成
# "日产日期"），宽松首字 + 后面的"取最完整值"双保险。
_RE_PRODUCTION = re.compile(r"(?:[生先土王日牛]?产日期|[生先土王日牛]?产期|出厂日期|灌装日期|包装日期|制造日期|MFG|MFD|PRODUCT|PROD|PRODUCTION|MANUFACTURED?|MANUFACTURING)\s*(?:DATE|DAY|DT)?\s*[:：]?\s*(\d{8}|\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?)", re.I)
_RE_SHELF_DURATION = re.compile(r"(?:保质期|保质[重事其]|保质日期|保质期限|有效期|保存期|保鲜期|最佳食用期)[:：]?\s*(?:常温|阴凉|冷藏|冷冻|避光|干燥|通风|未开封|未启封|条件下|环境|保存|贮藏|储存|存放|于|处|请){0,6}\s*(\d{1,4}|[一二两三四五六七八九十]{1,3})\s*(个月|月|天|日|周年|年|周)")
_RE_LOOSE_DATE = re.compile(_DATE_CORE)
_RE_COMPACT = re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")  # 20240115
# 点阵喷码（进口商品常见）：PD = production date，ED = expiry date。
# 日期值常为空格分隔（"PD: 2015 03 06"），普通分隔符正则覆盖不到。
_DOT_DATE_VALUE = r"(\d{4}[-/.年\s]\d{1,2}[-/.年\s]\d{1,2}|\d{4}[-/.年]\d{1,2}|\d{8})"
_RE_DOT_PROD = re.compile(r"\bPD\b\s*[:：]?\s*" + _DOT_DATE_VALUE, re.I)
_RE_DOT_EXP = re.compile(r"\bED\b\s*[:：]?\s*" + _DOT_DATE_VALUE, re.I)
# 空格分隔日期："2015 03 06"（点阵喷码拆行后最常见的形态）
_RE_SPACE_DATE = re.compile(r"(?<!\d)(20\d{2})\s+(0[1-9]|1[0-2])\s+(0[1-9]|[12]\d|3[01])(?!\d)")
# 非生产性日期关键词：这类日期不能当作生产日期（上市≠生产）
_RE_NON_PROD_KEYWORD = re.compile(r"(?:上市|上架|促销|抢购|现货|到店|到货)日期?")
# 日期"指向其他位置"的印刷提示：见包装背面 / 详见侧面 / 见喷码 等。
# OCR 常把关键词与提示拆成多行、中间隔上营养成分表等杂行（实测隔约 40 字），
# 容忍度放到 60 字；同时要求"见"与位置词相邻，避免误报。
_RE_DATE_ELSEWHERE = re.compile(
    r"(?:日期|保质期|有效期)[^。；\n]{0,60}?(?:见|详见|请见|见印|印刷)"
    r"[^。；\n]{0,8}(?:背面|反面|侧面|底部|瓶盖|瓶底|罐盖|罐底|盒盖|盒底|喷码|袋内|标签|封口|打印)"
    r"|(?:见|详见|请见)[^。；\n]{0,6}(?:包装|背面|反面|侧面|底部|瓶盖|瓶底|罐盖|罐底|盒盖|盒底|喷码|袋内|标签|封口|打印)")


def _parse_ymd(tok: str) -> Optional[dt.date]:
    """把 2024-01-15 / 2024.1.5 / 2024年1月15日 / 2024.03 / 2015 03 06 解析为 date。"""
    tok = _fix_digits(tok)
    m = re.match(r"^(\d{4})[-/.年\s](\d{1,2})(?:[-/.月\s](\d{1,2})日?)?$", tok)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), m.group(3)
    if d is None:                      # 仅年月：生产取月初，到期取月末由调用方决定
        if not (1 <= mo <= 12):
            return None
        return _mk(y, mo, 1)
    return _mk(y, mo, int(d))


_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_to_int(tok: str) -> Optional[int]:
    """中文数字 -> int（支持 0-999：三年、十八个月、二十五天）。"""
    tok = tok.strip()
    if not tok:
        return None
    if tok.isdigit():
        return int(tok)
    if "百" in tok:
        head, _, rest = tok.partition("百")
        n = (_CN_DIGITS.get(head, 1) if head else 1) * 100
        return n + (_cn_to_int(rest) or 0)
    if "十" in tok:
        head, _, rest = tok.partition("十")
        n = (_CN_DIGITS.get(head, 1) if head else 1) * 10
        return n + (_CN_DIGITS.get(rest, 0) if rest else 0)
    if len(tok) == 1:
        return _CN_DIGITS.get(tok)
    return None


def _non_production_dates(text: str) -> set:
    """找出紧跟上市/上架等非生产关键词之后的日期值（这些不是生产日期）。

    OCR 放大变体会把「上市日期20240702」拆成多行，仅靠"前6字符"判断不够，
    这里做语义级排除：该关键词后面的日期值，在任何位置出现都不作为生产日期。
    """
    out = set()
    for m in _RE_NON_PROD_KEYWORD.finditer(text):
        tail = text[m.end():m.end() + 20]
        dm = re.search(r"\d{8}", tail)
        if dm:
            d = _mk(int(dm.group()[:4]), int(dm.group()[4:6]), int(dm.group()[6:8]))
            if d:
                out.add(d)
            continue
        dm2 = re.search(_DATE_CORE, tail)
        if dm2:
            d = _parse_date_token(dm2.group(0))
            if d:
                out.add(d)
    return out


def _parse_date_token(tok: str) -> Optional[dt.date]:
    """解析日期 token：支持 2024-01-15 / 2024.1.5 / 2024年1月15日 / 2024.03 / 20240115。"""
    tok = _fix_digits(tok)
    if re.fullmatch(r"\d{8}", tok):
        return _mk(int(tok[:4]), int(tok[4:6]), int(tok[6:8]))
    return _parse_ymd(tok)


# ---------------------------------------------------------------- 解析结果

@dataclass
class DateParseResult:
    production_date: Optional[dt.date] = None
    shelf_life_days: Optional[int] = None
    expiry_date: Optional[dt.date] = None      # 显式到期日（保质期至）
    matched_texts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return bool(self.production_date or self.shelf_life_days or self.expiry_date)


def _token_has_day(tok: str) -> bool:
    """日期 token 是否含"日"部分。

    紧凑 8 位（20220117）恒有；其余按数字组数：三段（2022年1月17日 /
    2022-01-17 / "2016 03 05"）有"日"，两段（2022年1月 / "2016 03"）没有。
    """
    fixed = _fix_digits(tok).strip()
    if re.fullmatch(r"\d{8}", fixed):
        return True
    return len(re.findall(r"\d+", fixed)) >= 3


def _pick_date_token(pattern, text: str):
    """同一关键词的多次匹配中取最完整的日期值，返回 (token, date) 或 None。

    OCR 对同一物理行常给出多个部分读数（实测蛋黄桃酥标签：
    "产日期：2022年1月" / "日：2022年1月1" / "期：2022年1月17日"），
    旧逻辑取第一个匹配，截断值把 17 日读成 1 日（2022-01-17 → 2022-01-01）。
    优先取含"日"的完整值，其次取 token 最长者（截断行必是真值的前缀）。
    """
    best = None
    best_key = None
    for m in pattern.finditer(text):
        tok = m.group(1)
        d = _parse_date_token(tok)
        if not d:
            continue
        fixed = _fix_digits(tok)
        has_day = _token_has_day(tok)
        key = (1 if has_day else 0, len(fixed))
        if best_key is None or key > best_key:
            best, best_key = (tok, d), key
    return best


def _upgrade_partial(tok: str, d, text: str):
    """关键词行只读出年月（OCR 把一行拆成多个部分读数）时，全文找
    同年同月但带"日"的完整值升级。

    实测蛋黄桃酥标签"生产日期：2022年1月17日"被 OCR 拆成三行：
    '产日期：2022年1月' / '日：2022年1月1' / '期：2022年1月17日'，
    关键词只够到第一行（2022年1月 → 2022-01-01，17 日丢了）。
    同一年月的完整读数必是同一物理日期的另一种读法，升级安全；
    找不到完整值则维持原值（确实是只印年月的包装）。
    """
    if d is None or _token_has_day(tok):
        return d
    for m in _RE_LOOSE_DATE.finditer(text):
        d2 = _parse_date_token(m.group(0))
        if d2 and d2.year == d.year and d2.month == d.month and d2.day != 1:
            return d2
    for m in _RE_SPACE_DATE.finditer(text):     # "2015 03 06" 拆行形态
        d2 = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d2 and d2.year == d.year and d2.month == d.month and d2.day != 1:
            return d2
    return d


def parse_date_text(raw_text: str) -> DateParseResult:
    """从 OCR 文本中抽取日期与保质期信息。"""
    res = DateParseResult()
    if not raw_text or not raw_text.strip():
        return res
    text = normalize_text(raw_text)

    # 1) 显式到期日
    expiry_token = None
    picked = _pick_date_token(_RE_EXPIRY_TO, text)
    if picked:
        expiry_token = picked[0]
        res.matched_texts.append(expiry_token)
    if expiry_token is None:
        picked = _pick_date_token(_RE_EN_EXP, text)      # EXP DATE / BEST BEFORE / USE BY
        if picked:
            expiry_token = picked[0]
            res.matched_texts.append(expiry_token)
    if expiry_token is None:
        picked = _pick_date_token(_RE_DOT_EXP, text)     # 点阵喷码 ED:
        if picked:
            expiry_token = picked[0]
            res.matched_texts.append(expiry_token)
    if expiry_token:
        d0 = _parse_date_token(expiry_token)
        if d0:
            d = _upgrade_partial(expiry_token, d0, text)
            # 原 token 仅年月、且没能升级出完整日期 -> 视为"保质期至该月底"
            if not _token_has_day(expiry_token) and d == d0:
                d = _end_of_month(d)
            res.expiry_date = d

    # 2) 生产日期（多处匹配取最完整值，见 _pick_date_token）
    prod_token = None
    picked = _pick_date_token(_RE_PRODUCTION, text)
    if picked:
        prod_token = picked[0]
        res.matched_texts.append(prod_token)
    if prod_token is None:
        picked = _pick_date_token(_RE_EN_PROD, text)     # MFG DATE / PRODUCT DATE（含空格日期）
        if picked:
            prod_token = picked[0]
            res.matched_texts.append(prod_token)
    if prod_token is None:
        picked = _pick_date_token(_RE_DOT_PROD, text)    # 点阵喷码 PD:
        if picked:
            prod_token = picked[0]
            res.matched_texts.append(prod_token)
    if prod_token:
        d0 = _parse_date_token(prod_token)
        if d0:
            d = _upgrade_partial(prod_token, d0, text)
            res.production_date = d
            if d > dt.date.today() + dt.timedelta(days=1):
                res.warnings.append("生产日期晚于今天，请人工核对")

    # 3) 保质期时长（支持"12个月"与"三年"等中文数字写法）
    # 同一包装常多处印保质期，OCR 偶有丢字（实测红豆奶茶"12个月"在另一处被
    # 读成"2个月"）。不取第一个命中，而取折算天数最大的读数：丢字只会让数字
    # 偏小，取大值更能还原真实保质期。
    dur = None
    best_days = None
    for m in _RE_SHELF_DURATION.finditer(text):
        n = _cn_to_int(m.group(1))
        if n is None:
            continue
        unit = m.group(2)
        res.matched_texts.append(m.group(0))
        if n <= 0 or n > 3650:
            res.warnings.append(f"保质期时长异常：{n}{unit}")
            continue
        if unit in ("个月", "月"):
            cand, approx = ("months", n), n * 30
        elif unit in ("天", "日"):
            cand, approx = ("days", n), n
        elif unit in ("年", "周年"):
            cand, approx = ("months", n * 12), n * 12 * 30
        elif unit == "周":
            cand, approx = ("days", n * 7), n * 7
        else:
            continue
        if best_days is None or approx > best_days:
            best_days, dur = approx, cand
    if dur:
        if dur[0] == "months" and res.production_date:
            res.expiry_date = res.expiry_date or add_months(res.production_date, dur[1])
            res.shelf_life_days = res.shelf_life_days or (res.expiry_date - res.production_date).days
        elif dur[0] == "days" and res.production_date:
            res.expiry_date = res.expiry_date or add_days(res.production_date, dur[1])
            res.shelf_life_days = res.shelf_life_days or dur[1]
        else:
            res.shelf_life_days = dur[1] * 30 if dur[0] == "months" else dur[1]

    # 4) 松散日期：无关键词的 YYYY-MM-DD / 20240115，通常印在包装上的是生产日期
    #    但"上市/上架日期"不是生产日期，须排除（语义级：该值在任何位置都排除）
    if res.production_date is None and res.expiry_date is None:
        excluded = _non_production_dates(text)
        loose = None
        saw_non_prod = False
        best_key = None

        def consider(tok: str, d0) -> None:
            """候选值 PK：含"日"的完整值 > 截断值；同分取 token 长者。
            OCR 对同一行常给出多个部分读数（"2022年1月" / "2022年1月1" /
            "2022年1月17日"），旧逻辑取第一个，17 日被读成 1 日。"""
            nonlocal loose, best_key
            if not d0:
                return
            fixed = _fix_digits(tok)
            has_day = bool(re.fullmatch(r"\d{8}", fixed)) or \
                bool(re.search(r"[-/.月]\s*\d{1,2}\s*日?", fixed))
            key = (1 if has_day else 0, len(fixed))
            if best_key is None or key > best_key:
                loose, best_key = d0, key
                res.matched_texts.append(tok)

        for m in _RE_LOOSE_DATE.finditer(text):
            d0 = _parse_date_token(m.group(0))
            if d0 and d0 in excluded:
                saw_non_prod = True
                continue
            if _RE_NON_PROD_KEYWORD.search(text[max(0, m.start() - 6):m.start()]):
                saw_non_prod = True
                continue
            consider(m.group(0), d0)
        if loose is None:
            for m in _RE_COMPACT.finditer(text):
                d0 = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if d0 and d0 in excluded:
                    saw_non_prod = True
                    continue
                if _RE_NON_PROD_KEYWORD.search(text[max(0, m.start() - 6):m.start()]):
                    saw_non_prod = True
                    continue
                consider(m.group(0), d0)
        if loose is None:
            for m in _RE_SPACE_DATE.finditer(text):   # "2015 03 06" 点阵喷码拆行形态
                d0 = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if d0 and d0 in excluded:
                    saw_non_prod = True
                    continue
                if _RE_NON_PROD_KEYWORD.search(text[max(0, m.start() - 6):m.start()]):
                    saw_non_prod = True
                    continue
                consider(m.group(0), d0)
        if loose:
            if loose > dt.date.today() + dt.timedelta(days=1):
                res.warnings.append("识别到的日期晚于今天，请人工核对")
            res.production_date = loose
        elif res.shelf_life_days is None:
            # 什么都没解析到：给出"去哪里找日期"的针对性提示
            if saw_non_prod:
                res.warnings.append("仅识别到上市/上架日期（非生产日期），未找到生产日期与保质期，请手动填写")
            else:
                hint = date_elsewhere_hint(text)
                res.warnings.append(hint or "未找到生产日期/保质期，可能印在包装其他位置（侧面/底部/背面），请翻面拍摄或手动填写")

    return res


def date_elsewhere_hint(text: str) -> Optional[str]:
    """文本中是否有"日期印在其他位置"的印刷提示（见背面/见侧面/见喷码等）。"""
    if not text:
        return None
    if _RE_DATE_ELSEWHERE.search(normalize_text(text)):
        return ("包装上注明生产日期/保质期印在其他位置（背面/侧面/底部等），"
                "请翻面对准该区域拍摄，或手动填写")
    return None


def suggest_expiry(production_date: Optional[dt.date],
                   shelf_life_days: Optional[int],
                   category_default_days: int,
                   expiry_date: Optional[dt.date] = None,
                   today: Optional[dt.date] = None):
    """
    综合解析结果 + 类别默认值，给出最终 (生产日期, 保质期天数, 到期日期, 说明)。
    优先级：显式到期日 > 生产日期+时长 > 生产日期+类别默认 > 仅时长(按今天估算)。
    """
    today = today or dt.date.today()
    prod, days, expiry, notes = production_date, shelf_life_days, expiry_date, []

    if expiry is None and shelf_life_days and prod is None:
        # 仅时长：按今天生产估算（机器建议，需用户确认）
        prod, expiry = today, add_days(today, shelf_life_days)
        notes.append("仅识别到保质期时长，按'今天生产'估算，请人工核对")
    if expiry is None and shelf_life_days and prod:
        expiry = add_days(prod, shelf_life_days)
    if expiry is None and prod and category_default_days:
        expiry = add_days(prod, category_default_days)
        days = days or category_default_days
        notes.append(f"未识别到保质期，按类别默认 {category_default_days} 天估算")
    if expiry is None and not prod and not shelf_life_days:
        # 兜底文案统一由 parse_date_text 的三级精准提示给出，避免重复拼接
        pass

    if expiry and expiry < today:
        notes.append("该物品按识别结果已过期，请核对")
    return prod, days, expiry, notes


# ---------------------------------------------------------------- 临期分档

def expiry_status(expiry_date: Optional[dt.date], today: Optional[dt.date] = None,
                  near_days: int = 7):
    """返回 (状态, 剩余天数)：normal / near / expired / unknown。"""
    if not expiry_date:
        return "unknown", None
    today = today or dt.date.today()
    remaining = (expiry_date - today).days
    if remaining < 0:
        return "expired", remaining
    if remaining <= near_days:
        return "near", remaining
    return "normal", remaining
