"""
识别流水线编排：拍照 -> 检测 -> 归类 -> 日期抽取 -> 临期计算（端到端数据流）。

三条识别通道互补：
  1. YOLO11n 视觉检测（COCO 80 类，认得出香蕉/瓶子的通用外形）；
  2. RapidOCR 包装文字（品类关键词 + 生产日期/保质期正则）；
  3. CLIP 零样本图文匹配（YOLO 认不出、包装没文字时的视觉初判）。

检测不到目标时，退而按"包装日期文字区域"的空间聚类生成草稿，
最后经 _consolidate_drafts 清洗合并：去掉条码/配料表等噪声聚类，
把日期草稿与名称草稿合并成一个完整草稿。
"""
import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from sqlalchemy.orm import Session

from .. import config
from ..models import Category, CorrectionMemory, Setting
from ..schemas import DetectionDraft, Candidate
from . import detector, ocr, clip_cls, dotmatrix, region_cls
from .date_parser import (parse_date_text, suggest_expiry, normalize_text,
                          date_elsewhere_hint, _parse_date_token)

# ---------------------------------------------------------------- 类别解析

# 包装文字关键词 -> (物品名, 类别, 默认保质期天数)。
# 包装印字是比 COCO 视觉更可靠的品类证据：YOLO11n 只认识 80 类通用物体，
# 对盒/袋/罐装食品大量误检（如实测羊奶盒被误检为 book），此时以文字为准。
_KEYWORD_HINTS = [
    (re.compile(r"羊奶|羊乳|鲜羊乳"), "鲜羊奶", "食品", 7),
    # 酸奶/黄油/奶酪须排在"牛奶"前：酸奶盒配料印"生牛乳"、黄油盒配料印
    # "巴氏杀菌乳"，都会被"牛乳"抢先命中成纯牛奶（羊奶当初就是因此
    # 调到最前的，另几个乳制品同理）
    (re.compile(r"酸奶|发酵乳|风味发酵"), "酸奶", "食品", 14),
    (re.compile(r"黄油|奶油"), "黄油", "食品", 90),
    (re.compile(r"奶酪|芝士|干酪|再制奶酪"), "芝士/奶酪", "食品", 30),
    # 注意：不能把"巴氏杀菌"（工艺词）当牛奶关键词——包装印"巴氏杀菌乳"
    # 只是说明杀菌方式，任何乳畜奶都会有。实测鲜羊奶包装印"巴氏杀菌乳"，
    # 该工艺词 4 字胜过"羊奶"2 字，把羊奶误判成"纯牛奶"。
    (re.compile(r"纯牛奶|鲜牛奶|牛奶|牛乳|灭菌乳|调制乳"), "纯牛奶", "食品", 7),
    (re.compile(r"火腿|香肠|培根|腊肉"), "火腿/香肠", "食品", 45),
    # 曲奇/饼干须排在"面包"前：包装常把"曲奇饼干"与"糕点"混印，先命中细分项
    (re.compile(r"曲奇|饼干|酥饼|酥性|Cookies|COOKIES|BISCUIT|CRACKER"), "曲奇/饼干", "食品", 365),
    # 桃酥/中式糕点：散装称重标签的标题行（实测"蛋黄桃酥称斤"，配料表
    # 印"植物油"曾被误判成食用油——名称证据已排除配料区块）
    (re.compile(r"桃酥|酥糖|云片糕|绿豆糕|桂花糕|凤梨酥"), "桃酥/糕点", "食品", 365),
    # 坚果/炒货：须排在"曲奇/饼干"后、避免与"糕点"交叉；配料表常见"腰果、食用盐"
    (re.compile(r"腰果|核桃|杏仁|开心果|榛子|夏威夷果|碧根果|瓜子|花生|蚕豆|炒货|坚果|每日坚果|Cashew|CASHEW|WALNUT|ALMOND|NUTS"), "坚果/炒货", "食品", 180),
    (re.compile(r"巧克力|可可|糖果|奶糖|软糖"), "糖果/巧克力", "食品", 365),
    (re.compile(r"咖啡|拿铁|美式|咖啡豆"), "咖啡", "食品", 540),
    # 冷冻浆果须排在"冷冻水饺"前：包装常把"冷冻混合莓果"与速冻食品混放，
    # 先命中细分项（实测购物车照片读出"混合莓果 FROZEN MIX BERRY"）
    (re.compile(r"莓果|浆果|蓝莓|草莓|树莓|黑莓|桑葚|混合莓|BERRY|BERRIES"), "浆果/莓果", "食品", 365),
    (re.compile(r"椰子水|椰子汁|椰水|Coconut|COCONUT"), "椰子水", "食品", 365),
    # 榴莲千层/班戟等鲜糕点的保质期按"蛋糕"算；裸"榴莲"也归蛋糕
    # （购物车实测盖碗榴莲千层 OCR 成"榴莲干"，取宽匹配后用户可改）
    (re.compile(r"榴莲|千层|班戟|拿破仑"), "蛋糕", "食品", 3),
    # 梨系饮品：裸"梨"不匹配——"凤梨/菠萝"是水果不是梨汁（实测负例）
    (re.compile(r"梨汁|软儿梨|雪梨|秋梨|鸭梨|香梨|蜜梨|梨肉|梨丁"), "果汁", "食品", 365),
    (re.compile(r"面包|吐司|糕点|烘焙"), "面包", "食品", 5),
    (re.compile(r"蛋糕"), "蛋糕", "食品", 3),
    (re.compile(r"果汁|果蔬汁"), "果汁", "食品", 365),
    (re.compile(r"可乐|汽水|碳酸饮料"), "可乐/汽水", "食品", 365),
    (re.compile(r"矿泉水|饮用纯净水|纯净水|天然水"), "矿泉水", "食品", 730),
    (re.compile(r"奶茶|奶绿|奶青"), "奶茶", "食品", 180),
    (re.compile(r"意大利面|意面|通心粉|螺丝面|螺旋面|蝴蝶面|SPAGH|SPAGHETTI|PENNE|MACARONI"), "意大利面", "食品", 730),
    # 注意：不能匹配裸"面条"——烹饪说明"加入适量面条"不是产品名（实测误判）
    (re.compile(r"挂面|拉面|刀削面|乌冬面|鸡蛋面|阳春面"), "挂面/面条", "食品", 365),
    (re.compile(r"红茶|乌龙|普洱|铁观音|大红袍|龙井|碧螺春|毛尖|白茶|花茶|茉莉.*茶|茶叶|茶包|锡兰|三炮台|盖碗茶"), "红茶", "食品", 730),
    (re.compile(r"方便面|泡面"), "方便面", "食品", 180),
    (re.compile(r"罐头"), "罐头", "食品", 730),
    (re.compile(r"大米|稻米|东北米"), "大米", "食品", 365),
    # 酱类须排在"酱油/调味"前：拌饭酱/辣酱等更具体，且"酱油"含"酱"字不能抢匹配
    (re.compile(r"辣酱|拌饭酱|辣椒酱|黄豆酱|甜面酱|香其酱|牛肉酱|海鲜酱|蒜蓉酱|豆豉酱|拌饭|GOCHUJANG|GOCHUJANG|豆瓣酱|火锅蘸酱"), "辣酱/酱类", "食品", 540),
    (re.compile(r"食用油|植物油|菜籽油|大豆油|花生油|玉米油"), "食用油", "食品", 540),
    # 裸"调味"不能当关键词：标签"使用范围"常印"可做调味剂/调味用"，
    # 是功能用途描述不是产品名（实测维生素C 标签被误判成酱油/调味）。
    # 只有"调味料/调味品/调味酱/调味粉/调味汁/调味油"这类成品名才命中。
    (re.compile(r"酱油|调味(?:料|品|酱|粉|汁|油)|蚝油|番茄酱|沙拉酱"), "酱油/调味", "食品", 540),
    # 生鲜肉类须排在"方便面/火腿"后：方便面口味名常含"牛肉/猪肉"
    # （"红烧牛肉面"是方便面不是生肉——实测误判）
    (re.compile(r"牛肉|安格斯|谷饲|肥牛|牛排|猪肉|鸡肉|羊肉"), "生鲜肉类", "食品", 3),
    # 蔬菜须排在"番茄酱"后："番茄酱"含"番茄"；"水果菜"（果蔬包）归蔬菜不归水果
    (re.compile(r"芦笋|芹菜|菠菜|生菜|油菜|茼蒿|黄瓜|番茄|西红柿|水果菜|鲜蔬|有机蔬菜"), "蔬菜", "食品", 5),
    (re.compile(r"柚子|橙柚|葡萄柚|西柚|柠檬|蜜柚|水果"), "水果", "食品", 7),
    (re.compile(r"水饺|速冻|汤圆|包子"), "冷冻水饺", "食品", 180),
    (re.compile(r"冰淇淋|雪糕|冰棒"), "冰淇淋", "食品", 365),
    (re.compile(r"感冒|颗粒剂|胶囊|退烧|消炎药"), "感冒药", "药品", 730),
    (re.compile(r"维生素|钙片|保健品|益生菌"), "维生素", "药品", 730),
    (re.compile(r"创可贴|创口贴|医用敷贴"), "创可贴", "药品", 1095),
    (re.compile(r"洗发水"), "洗发水", "日化", 730),
    (re.compile(r"沐浴露"), "沐浴露", "日化", 730),
    (re.compile(r"牙膏"), "牙膏", "日化", 730),
    (re.compile(r"口罩"), "口罩", "日化", 730),
    (re.compile(r"洗衣液|洗洁精|消毒液"), "洗衣液", "日化", 730),
]

# 非牛乳畜奶：出现这些词时放弃泛称"纯牛奶"，让位给具体的乳品条目
# （实测鲜羊奶包装同时印"巴氏杀菌乳"，被泛称抢走名称）
_RE_OTHER_MILK = re.compile(r"羊奶|羊乳|驼奶|驼乳|马奶|水牛奶|牦牛奶")


# 配料表区块：GB 标签上"食品名称"之下是配料表，常印"植物油/黄油/鸡蛋/
# 小麦粉"等原料名。名称证据若不过滤，蛋黄桃酥会被配料里的"植物油"
# 判成"食用油"（实测）。起始行 OCR 常把"配料表"读成"配表"。
_RE_INGREDIENT_START = re.compile(r"配料|配表|配枓|原料|成分表")
# 顿号/逗号/句号分隔的原料串（配料表被 OCR 拆行后的续行）
_RE_INGREDIENT_WORD = re.compile(
    r"植物油|白砂糖|小麦粉|食用盐|食盐|鸡蛋|黄油|奶油|芝士|奶酪|奶粉|麦芽|糊精|"
    r"添加剂|砂糖|面粉|淀粉|氢化|精炼|棕榈|蔗糖|葡萄糖|食用胶|山梨酸钾|"
    # 饮料/奶茶类配料续行常见词（实测红豆奶茶"…速溶红茶粉，脱脂乳粉…"这行
    # 是配料表续行却不含"配料："，未被判为配料，导致"红茶"被当成独立物品）
    r"乳粉|炼乳|茶粉|植脂末|香精|香料|乳化剂|增稠剂|甜味剂|防腐剂|二氧化硅|"
    r"抗结剂|酸度调节剂|色素|稳定剂|麦芽糊精")


def _ingredient_flags(texts: List[str]) -> List[bool]:
    """逐行判断是否为配料表行（名称证据需排除这些行）。

    无状态规则——不按阅读顺序推断区块：① 配料起头行（配料/原料/成分表）；
    ② 含分隔符且含原料名词的行（配料拆行续行，如"…马蛋、植物油"）。
    实测两种边界：蛋黄桃酥的"…植物油"必须排除（否则误判食用油）；
    意大利面包装的品名"意大利面1"恰好印在配料区 y 范围内，顺序推断
    会把它吞掉导致关键词丢失——所以只认原料串特征，不吞邻近行。
    """
    flags: List[bool] = []
    for t in texts or []:
        s = normalize_text(t)
        is_ing = bool(_RE_INGREDIENT_START.search(s)) or \
            (any(c in s for c in "、，。") and bool(_RE_INGREDIENT_WORD.search(s)))
        flags.append(is_ing)
    return flags


def _name_evidence_texts(texts: List[str]) -> List[str]:
    """名称匹配专用的文本行：剔除配料表区块（配料原料名不是产品名）。"""
    flags = _ingredient_flags(texts)
    return [t for t, f in zip(texts or [], flags) if not f]


# 售卖后缀：散装称重标签把品名与"称斤/16.00元"印在同一行
_RE_SALES_SUFFIX = re.compile(r"(?:称斤|秤斤|称重|散装|计量称重|单价|特价|促销)+$")


def _clean_product_name(name: Optional[str]) -> Optional[str]:
    """剥掉产品名行尾的售卖信息（"蛋黄桃酥称斤" -> "蛋黄桃酥"）。"""
    if not name:
        return name
    return _RE_SALES_SUFFIX.sub("", name) or name


def _specific_name(prod_name: Optional[str], hint: Optional[tuple]) -> Optional[str]:
    """产品名行是否比关键词类名更具体（包含类名的组成部分）。

    关键词给出的是类名（"桃酥/糕点"），包装标题行是具体产品
    （"蛋黄桃酥"含"桃酥"）——类名只用来定类别，名称用具体的。
    """
    if not prod_name or not hint:
        return None
    parts = set([hint[0]] + _name_parts(hint[0]))
    return prod_name if any(p in prod_name for p in parts) else None


def _is_class_name(name: Optional[str]) -> bool:
    """名称是否只是关键词类名（"辣酱/酱类"），而非包装上的具体产品名。

    类名集合 = 关键词类名 + 词典条目名。日期聚类读到关键词但拿不到
    产品名时名称会是类名（实测拌饭酱：日期在顶部、大字品名在另一区），
    此时应尝试升级为具体产品名。
    """
    if not name:
        return False
    classes = {h[1] for h in _KEYWORD_HINTS}
    classes.update(c[0] for c in config.CANDIDATE_DICTIONARY)
    return name in classes


def _ocr_hint(texts: List[str]) -> Optional[tuple]:
    """从 OCR 文本中匹配包装关键词，返回 (物品名, 类别名, 默认天数) 或 None。

    先剔除配料表区块：配料里的"植物油/黄油"是原料不是产品
    （实测蛋黄桃酥标签被误判成食用油）。

    取"最长匹配"而非首个命中：短泛词极易被正文/用途说明里的字面命中
    （实测维生素C 标签"可做调味剂"命中裸"调味"）。匹配片段越长说明越
    具体（"维生素"3字胜过"调味"2字）；同长度按关键词表顺序（具体品名
    排在泛类名之前）。
    """
    if not texts:
        return None
    joined = normalize_text(" ".join(_name_evidence_texts(texts)))
    # 非牛乳畜奶（羊/驼/马/水牛）出现时，泛称"纯牛奶"一定不准（羊奶包装也常
    # 印"调制乳/灭菌乳"），让位给具体的乳品条目。
    other_milk = bool(_RE_OTHER_MILK.search(joined))
    best_len = -1
    best = None
    for pat, name, cname, days in _KEYWORD_HINTS:
        if name == "纯牛奶" and other_milk:
            continue
        m = pat.search(joined)
        if m and len(m.group(0)) > best_len:
            best_len = len(m.group(0))
            best = (name, cname, days)
    return best


# 预包装食品强制标注的"食品名称：XXX" / "产品名称：XXX"——比词典类名更准确，
# 优先提取。很多包装印"产品名称"而非"食品名称"（实测黄油卷印"产品名称：总统牌
# 淡味黄油卷"），两者都要覆盖。限制：2-12 字、不含数字（排除"食品名称：见包装"
# 等杂项）、不含其他标签关键词。
_RE_FOOD_LABEL = re.compile(r"(?:食品名称|产品名称)[:：]?\s*([^\s，。；、,;:：（(]{2,12})")
_RE_LABEL_STOP = re.compile(r"配料|净含量|生产许可证|产品标准|保质期|生产日期|贮存|食用方法|产地|生产商|委托方|规格|等级|商标|条形码|合格")


def _food_label_name(texts: List[str]) -> Optional[str]:
    """提取包装上"食品名称：XXX"标注的原名（如"带衣腰果"）。"""
    for t in texts or []:
        s = normalize_text(t)
        m = _RE_FOOD_LABEL.search(s)
        if not m:
            continue
        name = m.group(1).strip()
        if _RE_LABEL_STOP.search(name):
            continue
        return name
    return None


# 包装标签/产地/平台等非产品名词，提取"包装文字产品名"时跳过
_RE_NON_NAME_WORD = re.compile(
    r"净含量|原料|生产|保质|贮存|食用|产地|生产商|委托|规格|等级|商标|条形码|合格|配料|保存|开启|避光|阴凉|冷藏|冷冻|日期|进口|出口|盒马|品牌|售后|电话|地址|请|勿|开封|置于|如有|质量|标准|编号|净重|提示|说明|方法"
    # 营养成分表术语：这类行常被误当品名（实测红豆奶茶把"反式脂肪"读成
    # "一反式脂肪"当成了产品名）。它们只出现在营养表里，绝不可能是品名。
    r"|营养成分|反式脂肪|碳水化合物|膳食纤维|能量|蛋白质|脂肪|NRV|每100|千焦|千卡|项目"
    # 企业/机构/报刊名：包装背面常见"XX食品有限公司""XX商贸"，
    # OCR 常把公司名拆成"宇和留案"+"有限"两行，前半截会被误当商品名
    # （实测薯片背面：唯一"像品名"的行是公司名碎片"宇和留案"）
    r"|有限|公司|集团|商贸|实业|科技|贸易|超市|商场|市场|大厦|大楼|工业园|都市报|报社|新闻|厂"
    # 营销/实验文案：包装常贴"擦除实验/结果对比"类测试标签（实测拌饭酱），
    # 或印活动标语，都不是产品名
    r"|实验|对比|结果|擦除|测试|促销|活动|中奖|扫码|直播|网红|撩妹"
    # AI 生成图自带水印文字，用户转存这类图时会被 OCR 读成产品名
    r"|AI生成|AI绘画|AI制作|AI图片|人工智能生成"
    # AI 生成图水印常被 OCR 截断成"A生成/生成"等变体
    r"|AI|生成|WORKBUDDY|AIGC")
_RE_GEO_WORD = re.compile(
    r"越南|泰国|韩国|日本|美国|澳大利亚|新西兰|南非|智利|秘鲁|墨西哥|印度|法国|德国|意大利|西班牙|荷兰|比利时|挪威|冰岛|中国|国产|进口|限定")
_RE_EN_WORD = re.compile(r"[A-Za-z]{3,}")

# 企业/机构/地址线索：出现在某行的"紧邻行"时，该行多半是公司名/地址碎片，
# 不能当商品名（OCR 常把"XX食品有限公司"拆成"宇和留案"+"有限"两行）
_RE_COMPANY_LIKE = re.compile(
    r"有限|公司|集团|商贸|实业|科技|贸易|超市|商场|市场|大厦|大楼|工业园"
    r"|都市报|报社|新闻|地址|大道|大街|园区|工厂|经销|代理商|制造商")


def _text_is_barcode_heavy(texts: List[str], thresh: float = 0.4) -> bool:
    """文字是否条码/批号密集（=拍的是包装背面，没有可读品名）。

    实测薯片背面：整页都是条码、批号、"保质期：9个月""生产日期"这类标注，
    唯一"像品名"的行是公司名碎片"宇和留案"——这种情况下强行取名只会更糟，
    应当放弃取名、交给上层用视觉类别/候选让用户确认。
    """
    lines = [t for t in (texts or []) if t and t.strip()]
    if not lines:
        return False
    heavy = sum(
        1 for t in lines
        if re.fullmatch(r"[\d\s\-./#]+", t.strip())
        or len(re.findall(r"\d", t)) >= 8
    )
    return heavy / len(lines) >= thresh


# 数量/价格后缀：散装与简装标签把品名和数量印在同一行
# （"鸡蛋 30枚" / "椰子水 100%" / "曲奇 200g"），提取产品名时先剥掉
_RE_QTY_SUFFIX = re.compile(
    r"[\s:：]*(?:\d+(?:\.\d+)?\s*(?:枚|个|包|袋|盒|瓶|罐|块|片|粒|颗|支|条|克|千克|毫升|升|元|g|kg|ml|l|L|%))+\s*$")


def _product_name_from_texts(texts: List[str]) -> Optional[str]:
    """从区域 OCR 文字中提取包装产品名（关键词/标注/修正都落空时的兜底）。

    取第一条"像产品名"的文字行：剥掉数量后缀后 2-8 个字符、至少 2 个汉字、
    不含数字、不含标签/产地/品牌词、非中英混排（标语/说明）。
    实测购物车照片：椰子水盒首行汉字是「椰子水」，排除「越南」「原料进口」
    后才不会取错；软儿梨瓶取到「兰州软儿梨」；模拟货架「鸡蛋 30枚」取到「鸡蛋」。
    """
    # 条码/批号密集（包装背面）：没有可读品名，宁可不取也不取乱码
    if _text_is_barcode_heavy(texts):
        return None
    for i, t in enumerate(texts or []):
        window = " ".join((texts or [])[max(0, i - 1): i + 2])
        # 企业/机构/地址碎片不当商品名（"宇和留案" 下一行就是 "有限"）
        if _RE_COMPANY_LIKE.search(window):
            continue
        s = (t or "").strip().strip("。．.,，、;；:：·")
        s = _RE_QTY_SUFFIX.sub("", s).strip()
        if not s:
            continue
        cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
        if cjk < 2 or len(s) > 8 or any(ch.isdigit() for ch in s):
            continue
        if _RE_NON_NAME_WORD.search(s) or _RE_GEO_WORD.search(s):
            continue
        if _RE_EN_WORD.search(s):
            continue
        return s
    return None


def _correction_conflicts(corr_name: str, texts: List[str]) -> bool:
    """区域包装文字是否给出了与历史修正矛盾的产品证据。

    修正记忆按视觉类别（raw_name）整体生效，但同一类别可能对应不同产品
    （bottle 可能是矿泉水也可能是椰子水——实测购物车照片双瓶误判）。
    只有"矛盾的产品证据"（关键词命中 / 食品名称标注 / 干净的产品名行）
    才挡掉修正；纯乱码/纯英文无可矛盾，沿用历史修正。
    """
    if not corr_name:
        return False
    joined = normalize_text(" ".join(texts or []))
    if not joined or corr_name in joined:
        return False
    return bool(_ocr_hint(texts) or _food_label_name(texts)
                or _product_name_from_texts(texts))


def _name_parts(name: str) -> List[str]:
    """词典名按 "/" 拆成可独立匹配的部分（"曲奇/饼干" -> ["曲奇", "饼干"]）。

    包装印的是"注心曲奇饼干"，词典全名"曲奇/饼干"并不是其子串，
    拆开部分后才能互相命中（实测曲奇包装候选列表因此没有对口项）。
    """
    return [p for p in re.split(r"[/／]", name or "") if len(p) >= 2]


def _matched_parts(name: str, joined: str) -> List[str]:
    """词典名在文本中命中的部分：全名命中返回 [全名]，否则返回命中的部分列表。"""
    if not joined or not name:
        return []
    if name in joined:
        return [name]
    return [p for p in _name_parts(name) if p in joined]


def _category_by_name(db: Session, name: str) -> Optional[Category]:
    return db.query(Category).filter(Category.name == name).first()


def _correction_for(db: Session, raw_name: str) -> Optional[CorrectionMemory]:
    if not raw_name:
        return None
    return (db.query(CorrectionMemory)
            .filter(CorrectionMemory.raw_name == raw_name)
            .order_by(CorrectionMemory.last_used_at.desc())
            .first())


def _candidates_for(db: Session, prefer_category: Optional[Category],
                    raw_name: str = "", texts: Optional[List[str]] = None,
                    clip_names: Optional[List[str]] = None,
                    limit: int = 10) -> List[Candidate]:
    """低置信度时的候选列表：优先同类词典项 + 名称/OCR 文字/CLIP 视觉相关匹配。"""
    cats = {c.name: c for c in db.query(Category).all()}
    pool = list(config.CANDIDATE_DICTIONARY)
    raw_lower = (raw_name or "").lower()
    joined = normalize_text(" ".join(texts or []))
    clip_lower = {n.lower() for n in (clip_names or [])}

    def score(entry):
        name, cname, days = entry
        s = 0
        if prefer_category and cname == prefer_category.name:
            s += 10
        if raw_lower and (raw_lower in name.lower() or name in raw_lower):
            s += 5
        if joined:
            # OCR 文字命中 -> 最相关：全名 +20；按 "/" 拆出的部分命中（如
            # "曲奇饼干"文字命中"曲奇/饼干"）也能排到前面，命中部分越多分越高
            mp = _matched_parts(name, joined)
            if name in mp:
                s += 20
            elif mp:
                s += 13 + 2 * min(len(mp), 2)
        if name.lower() in clip_lower:     # CLIP 视觉初判命中 -> 次相关
            s += 12
        return -s

    pool.sort(key=score)
    out: List[Candidate] = []
    for name, cname, days in pool[:limit]:
        c = cats.get(cname) or cats.get("其他")
        if c:
            out.append(Candidate(name=name, category_id=c.id,
                                 category_name=c.name,
                                 default_shelf_life_days=days))
    return out


# ---------------------------------------------------------------- 截图识别

# 用户反馈问题时经常直接截 App 界面当图片上传。截图中混入的大量界面文字
# （候选按钮、表单标签、提示语）会污染 OCR 聚类，先识别并过滤掉。
# 注意：只能放"界面独有"的词——"生产日期/保质期"包装上本来就印，放进来的话
# 任何食品照片都会被误判成截图（实测意大利面原图即因此误判）。
_UI_MARKERS = ["待确认草稿", "候选物品", "单条入库", "并入本次保存", "识别区域",
               "日期来源", "可从下方候选点选", "点击采用", "请补全物品名称",
               "已保存", "待确认", "未检测到已知物品", "物品名称",
               # 表单区提示语（OCR 拆行后每行只剩片段，需按词命中）
               "请点击", "手动填写", "手动输入", "输入框", "包装上注明",
               "印在其他位置", "请翻面", "对准该区域拍摄"]
_UI_MARKERS_N = [normalize_text(m) for m in _UI_MARKERS]
# 界面文字行的正则特征（比纯子串更稳：OCR 会把全角/半角、括号写歪）
_UI_LINE_PATTERNS = [
    r"OCR[:：]",                                    # 草稿卡上的"OCR：..."提示行
    r"保质期\s*[(（]\s*天\s*[)）]",                   # 表单标签"保质期(天)"（括号可能全半角混排）
    r"未找到生产日期", r"可能印在包装其他位置", r"请翻面拍摄",
    r"侧面\s*[/／]\s*底部\s*[/／]\s*背面",
    # App 顶栏当前日期（截图上传时会被当成包装上的生产日期，实测误判）
    r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*[·・、]?\s*星期",
    r"^\s*待确认\s*$", r"^\s*采用\s*[)）]?\s*$", r"^\s*角认\s*$",
]
# 表单标签/候选 chips 的独立短行（截图中出现即丢弃；精确匹配，不影响包装长句）
_UI_EXACT_LINES = {"类别", "存放位置", "数量", "生产日期", "保质期(天)", "保质期 (天)",
                   "到期日期", "年/月/日", "年/月／", "年/月/", "日期来源", "日期来源：",
                   "人工填写", "未指定", "食品", "日化", "药品", "文具", "其他",
                   "待确认", "角认", "物品名称", "单条入库", "并入本次保存",
                   "识别区域", "已保存", "件", "个识别区域"}
_CHIP_NAMES = {name for name, _, _ in config.CANDIDATE_DICTIONARY}


def _looks_like_app_screenshot(texts: List[str]) -> bool:
    """OCR 文本命中 ≥2 个界面特征词时，判定为用户上传的 App 截图。"""
    if not texts:
        return False
    joined = normalize_text(" ".join(texts))
    return sum(1 for m in _UI_MARKERS_N if m in joined) >= 2


def _is_ui_line(s: str) -> bool:
    """判断一行 OCR 文本是否为界面文字（候选按钮/表单标签/提示语等）。"""
    if not s:
        return False
    return (any(m in s for m in _UI_MARKERS_N)
            or any(re.search(p, s, re.I) for p in _UI_LINE_PATTERNS)
            or s in _UI_EXACT_LINES or s in _CHIP_NAMES)


def _filter_ui_lines(ocr_items: List[Dict], is_screenshot: bool) -> List[Dict]:
    """截图场景下剔除界面文字行，只保留照片内容文字。"""
    if not is_screenshot:
        return ocr_items
    return [t for t in ocr_items
            if not _is_ui_line(normalize_text(t["text"]))]


def _ui_line_boxes(ocr_items: List[Dict], is_screenshot: bool) -> List[tuple]:
    """截图场景下被剔除的界面文字行的 bbox 列表（用于丢弃落在界面区的检测框）。"""
    if not is_screenshot:
        return []
    return [_bbox_of(t["box"]) for t in ocr_items
            if _is_ui_line(normalize_text(t["text"]))]


def _covered_by_boxes(bbox, boxes: List[tuple], thresh: float = 0.35) -> bool:
    """bbox 与任一已有草稿框的重叠面积占自身比例是否超过阈值。

    名称聚类补漏时使用：文字区域已被 YOLO/区域 CLIP 草稿覆盖的跳过，
    避免同一包装生成两条草稿（实测购物车照片的"椰子水"文字区
    正好落在瓶子检测框内）。
    """
    if not boxes:
        return False
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    for (bx1, by1, bx2, by2) in boxes:
        ox = max(0, min(x2, bx2) - max(x1, bx1))
        oy = max(0, min(y2, by2) - max(y1, by1))
        if ox > 0 and oy > 0 and (ox * oy) / area >= thresh:
            return True
    return False


def _contained_in(bbox: tuple, boxes: List[tuple], thresh: float = 0.9) -> bool:
    """bbox 是否基本被任一框完整包含（用于清理被视觉区域覆盖的冗余文字草稿）。"""
    if not boxes:
        return False
    x1, y1, x2, y2 = bbox
    area = max(1, x2 - x1) * max(1, y2 - y1)
    for (bx1, by1, bx2, by2) in boxes:
        ox = max(0, min(x2, bx2) - max(x1, bx1))
        oy = max(0, min(y2, by2) - max(y1, by1))
        if ox * oy >= thresh * area:
            return True
    return False


def _union_area_ratio(boxes: List[tuple], width: int, height: int) -> float:
    """所有框并集面积占整图比例的粗略估计（重叠不扣减，偏保守）。

    用于判断画面是否已被识别区域覆盖：覆盖比例低说明画面里大概率还有
    没被文字/检测框命中的物品（如散装果蔬），值得跑一次视觉切分补漏。
    """
    if not boxes or width <= 0 or height <= 0:
        return 0.0
    total = 0
    for (x1, y1, x2, y2) in boxes:
        total += max(0, x2 - x1) * max(0, y2 - y1)
    return min(1.0, total / (width * height))


def _covered_by_ui(bbox, ui_boxes: List[tuple], thresh: float = 0.6) -> bool:
    """检测框被界面文字行覆盖的面积比例是否超过阈值。

    截图右侧面板的表单/提示文字区常被 YOLO 误检成"物品"，
    但这些区域布满界面文字行，真实包装区域不会。
    """
    if not ui_boxes:
        return False
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    cover = 0
    for (ux1, uy1, ux2, uy2) in ui_boxes:
        ox = max(0, min(x2, ux2) - max(x1, ux1))
        oy = max(0, min(y2, uy2) - max(y1, uy1))
        if ox > 0 and oy > 0:
            cover += ox * oy
    return cover / area >= thresh


# ---------------------------------------------------------------- 主流程

_MAX_DRAFTS = 10          # 单张照片最多产出的草稿数（含补漏）
_MAX_SUPPLEMENT = 6       # YOLO 检出后，文字锚点最多补漏的草稿数


def recognize_photo(db: Session, image_path: Path, rel_path: str,
                    today: Optional[dt.date] = None) -> dict:
    """
    识别单张照片，返回 PhotoResult 字典（草稿未入库，由前端确认后批量保存）。
    """
    today = today or dt.date.today()
    pil = Image.open(image_path)
    pil.load()
    # 统一转 RGB：用户上传的图可能是 P(调色板)/RGBA/LA/CMYK 等模式
    # （P 模式 PNG 会让 ImageEnhance.Contrast 抛 "image has wrong mode"，
    #  导致整张图识别失败），YOLO/OCR/CLIP/点阵通道都按 RGB 处理。
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    # 统一降到最长边 MAX_IMAGE_SIZE 再进入识别：YOLO 本就内部缩到该尺度、
    # CLIP 裁剪会 resize 到 224，都不受影响；唯有 RapidOCR 与点阵解码此前按
    # 原图分辨率（OCR 还会再放大 1.5×）跑，手机 4000px 大图下 OCR 成了隐藏
    # 耗时大头。1280 对包装日期文字绰绰有余，零识别精度损失。YOLO/OCR/点阵
    # /CLIP 都基于同一张 pil，bbox 坐标天然一致，不会错位。
    if max(pil.size) > config.MAX_IMAGE_SIZE:
        scale = config.MAX_IMAGE_SIZE / max(pil.size)
        pil = pil.resize((int(pil.size[0] * scale), int(pil.size[1] * scale)),
                         Image.LANCZOS)
    width, height = pil.size

    # 1. 多目标检测
    detections = detector.detect(pil)

    # 2. 全文 OCR（用于包装小字定位与兜底）
    #    按 y 坐标排序成阅读顺序：日期关键词与"见包装背面"等提示常被 OCR
    #    拆成多行，不排序的话拼接文本里两者相隔甚远，语义匹配会失败。
    full_ocr = ocr.ocr(pil)
    full_ocr = sorted(full_ocr,
                      key=lambda t: (sum(p[1] for p in t["box"]) / 4,
                                     sum(p[0] for p in t["box"]) / 4))
    is_screenshot = _looks_like_app_screenshot([t["text"] for t in full_ocr])
    full_texts = [t["text"] for t in full_ocr]
    # 截图场景下，界面文字（候选按钮/表单标签/顶栏日期）不参与日期与名称匹配：
    # 实测候选 chip "芝士/奶酪"被当成物品名、顶栏"2026年10月13日"被当成生产日期。
    filtered_items = _filter_ui_lines(full_ocr, is_screenshot)
    match_texts = [t["text"] for t in filtered_items]

    # 2.5 点阵喷码兜底：点阵字符的墨点互不连通，RapidOCR 检测器对整行失效
    #     （实测黄油包装 PD:/ED: 行只读出"39 33"垃圾）。常规 OCR 读不到
    #     生产/到期日期时，绕过检测器直接读横条提取，重构文本并入全文。
    #     提示框取过滤后的 OCR 行（截图场景不扫界面文字区）。
    _parsed_all = parse_date_text(" ".join(match_texts))
    if not (_parsed_all.production_date and
            (_parsed_all.expiry_date or _parsed_all.shelf_life_days)):
        dot_items = dotmatrix.read_dotmatrix_dates(
            pil, hint_boxes=_date_hint_boxes(filtered_items))
        if dot_items:
            full_ocr = sorted(full_ocr + dot_items,
                              key=lambda t: (sum(p[1] for p in t["box"]) / 4,
                                             sum(p[0] for p in t["box"]) / 4))
            full_texts = [t["text"] for t in full_ocr]
            match_texts = [t["text"] for t in _filter_ui_lines(full_ocr, is_screenshot)]

    # 3. CLIP 零样本视觉初判（第三条识别通道，模型未安装时自动跳过）
    clip_top = clip_cls.classify(pil) or []
    clip_names = [c["name"] for c in clip_top[:3]]

    # 4. 每个检测目标 -> 草稿
    #    截图场景：YOLO 会把右侧面板的表单/提示文字区误检成"物品"，
    #    这些框布满界面文字行，直接丢弃，走 OCR 聚类兜底。
    ui_boxes = _ui_line_boxes(full_ocr, is_screenshot)
    drafts: List[DetectionDraft] = []
    for det in detections:
        if _covered_by_ui(det["bbox"], ui_boxes):
            continue
        # 复用整图 OCR 结果（full_ocr）取该框内文字，不再对每个检测框重跑
        # 完整 OCR——这是单张照片里最大的冗余耗时源（多物品照片尤为明显）。
        region_texts = ocr.texts_inside(pil, det["bbox"], pad=config.CROP_PAD,
                                        items=full_ocr)
        # 低置信通用容器类且框内读不到任何文字：背景误报，丢弃
        # （实测苹果照片 4 个 bottle 全是背景瓶状物；黄油盒虽被认成
        # bottle conf 0.73，但框内印着"黄油"，文字证据可用必须保留）
        if (det["class_name"] in _GENERIC_CLASSES
                and det["confidence"] < _GENERIC_CONF
                and not region_texts):
            continue
        texts = list(dict.fromkeys(region_texts))  # 保序去重
        date_text = " ".join(texts)
        # 易误判的通用类（book/bottle 等）用 CLIP 区域零样本分类纠正：
        # YOLO 的 COCO 80 类没有「膨化食品/薯片」等包装商品类，常把扁平零食
        # 袋误判成 book。裁剪该区域单独跑 CLIP，当命中与 COCO 映射不同的食品
        # /日化/药品类（score>=0.5）时注入 clip_hint 覆盖——正面薯片袋可 0.8+
        # 命中「薯片/膨化食品」；背面无视觉特征则 CLIP score 很低，不覆盖，
        # 仍走下方「请从候选中选择」兜底，不会退化成错误答案。
        if det["class_name"] in _MISLEAD_CLASSES:
            x1, y1, x2, y2 = det["bbox"]
            crop = pil.crop((max(0, x1), max(0, y1),
                             min(width, x2), min(height, y2)))
            regional = clip_cls.classify(crop, top_k=5) or []
            if regional and regional[0]["score"] >= 0.5:
                c = regional[0]
                mapped = config.COCO_CLASS_MAP.get(det["class_name"])
                mapped_name = mapped[1] if mapped else det["class_name"]
                if c["name"] != mapped_name:   # 仅当 CLIP 给出更具体的不同类别才覆盖
                    det = dict(det)
                    det["clip_hint"] = (c["name"], c["category"],
                                        c["default_shelf_life_days"])
        drafts.append(_build_draft(db, det, date_text, texts, today,
                                   all_texts=match_texts, clip_names=clip_names))

    # 4.5 文字锚点补漏：YOLO 只认 COCO 80 类，购物车/货架照片里大量包装
    #     （莓果袋、榴莲蛋糕、蔬菜包、牛肉托）检不出框，但 OCR 读得到
    #     产品名。按名称证据聚类补充草稿，与已有检测框重叠的跳过。
    #     实测购物车照片：2 个瓶子之外补出浆果/水果/生鲜肉类/蛋糕 4 条。
    supplement_note = ""
    if not is_screenshot and drafts and len(drafts) < _MAX_DRAFTS:
        extra = _drafts_from_ocr(db, full_ocr, today, is_screenshot,
                                 name_only=True,
                                 skip_boxes=[d.bbox for d in drafts if d.bbox])
        if extra:
            keep = extra[:_MAX_SUPPLEMENT]
            # 与已有草稿同名且自身无日期的补充草稿跳过（同一款包装的
            # 另一件，或检测框已覆盖的区域——实测拌饭酱两盒各出一条）
            exist_names = {d.name for d in drafts}
            keep = [d for d in keep
                    if not (d.name in exist_names
                            and not d.production_date
                            and not d.expiry_date)]
            drafts.extend(keep)
            supplement_note = (f"另按包装文字识别出 {len(keep)} "
                               f"个物品区域，请逐个确认")

    # 4.6 视觉补产（多物品场景）：画面里还有"无文字的生鲜/果蔬"时，即便已靠
    #     文字/瓶子识别出若干包装商品，也要把没有文字的物品切出来——否则购物车/
    #     菜篮照片里只有带标签的饮料被识别，番茄/柠檬等被整块漏掉（实测
    #     e701… 照片：2 个瓶子被识别，4 个果蔬区域全丢）。
    #     仅当已识别框未覆盖大部分画面时运行（单件物品照片不浪费 ~3s 推理）。
    if (not is_screenshot and config.MULTI_REGION_SPLIT
            and drafts and len(drafts) < _MAX_DRAFTS):
        exist_boxes = [d.bbox for d in drafts if d.bbox]
        if exist_boxes and _union_area_ratio(exist_boxes, width, height) < 0.55:
            regs = [r for r in region_cls.split_regions(pil)
                    if not _covered_by_boxes(r["bbox"], exist_boxes, thresh=0.5)]
            regs = regs[:_MAX_DRAFTS - len(drafts)]
            for i, reg in enumerate(regs, start=1):
                det = {"class_name": reg["name"],
                       "confidence": reg["score"],
                       "bbox": reg["bbox"],
                       "clip_hint": (reg["name"], reg["category"],
                                     reg["default_shelf_life_days"])}
                crop_texts = ocr.texts_inside(pil, reg["bbox"], pad=config.CROP_PAD,
                                             items=full_ocr)
                d = _build_draft(
                    db, det, " ".join(crop_texts), crop_texts, today,
                    all_texts=match_texts, clip_names=clip_names)
                drafts.append(_mark_region_draft(
                    db, d, reg["bbox"], i, len(regs), width, height))
                exist_boxes.append(reg["bbox"])
            if regs:
                region_note = ("另按视觉特征切分出无文字的物品区域"
                               "（果蔬/生鲜），名称与日期请逐个确认")

    # 4.7 冗余清理：被视觉切分区域完整覆盖、又无日期证据的文字/检测草稿删除，
    #     避免重复卡片。典型：OCR 从乱码行"非白橙柚"误匹配出"水果"卡片，
    #     而视觉切分已给出同一区域的水果卡片，二者重复。
    split_boxes = [d.bbox for d in drafts if d.region_split and d.bbox]
    if split_boxes:
        kept = []
        for d in drafts:
            if (not d.region_split and not d.production_date and not d.expiry_date
                    and d.bbox and _contained_in(d.bbox, split_boxes)):
                continue
            kept.append(d)
        drafts = kept

    # 5. 无检测结果：按 OCR 日期文字区域聚类生成草稿；仍无则给一条手动草稿
    fallback_used = False
    region_note = ""                  # 区域 CLIP 扫描的结果说明（多物品场景）
    if not drafts:
        fallback_used = True
        cluster_drafts = _drafts_from_ocr(db, full_ocr, today,
                                          is_screenshot=is_screenshot)
        # 5.1 名称锚点聚类：没有日期文字但读得到产品名时同样能生成草稿
        #     （购物车/货架远景照常无日期，靠名称补）
        name_drafts = _drafts_from_ocr(db, full_ocr, today, is_screenshot,
                                       name_only=True)
        if name_drafts:
            exist = [d.bbox for d in cluster_drafts if d.bbox]
            name_drafts = [d for d in name_drafts
                           if not _covered_by_boxes(d.bbox, exist)]
            # 与带日期的草稿同名且自身无日期时跳过：同一款包装的另一件
            # （或同一件的品名区），日期信息已由带日期的草稿承载
            # （实测拌饭酱：上盒只有喷码日期、下盒只有大字品名，各成一条）
            dated_names = {d.name for d in cluster_drafts
                           if d.production_date or d.expiry_date}
            name_drafts = [d for d in name_drafts
                           if not (d.name in dated_names
                                   and not d.production_date
                                   and not d.expiry_date)]
            cluster_drafts = cluster_drafts + name_drafts
        if cluster_drafts:
            drafts = cluster_drafts
        elif clip_top and clip_top[0].get("score", 0) >= 0.5:
            # 5.15 整图 CLIP 结论：照片主体单一（一个苹果/一瓶水）时直接
            #     以整图结论生成一条草稿。区域网格扫描在单物体照片上会把
            #     桌面、阴影、桌纹误判成多个物品（实测苹果照片扫出 8 条
            #     草稿，其中 5 条是"矿泉水/花瓶"类垃圾）。
            #     但整图判类不自信（< MULTI_SCENE_CONF）说明画面里根本不是
            #     一件东西——多件物品互相干扰才会把概率摊薄。此时改用
            #     互不重叠的 4x3 切分，一件物品一条草稿（见 split_regions）。
            c = clip_top[0]
            split: List[Dict] = []
            if (not is_screenshot and config.MULTI_REGION_SPLIT
                    and c["score"] < config.MULTI_SCENE_CONF):
                split = region_cls.split_regions(pil)
                if len(split) < 2:          # 只有一格达标 → 还是单件物品
                    split = []
            if split:
                for i, reg in enumerate(split, start=1):
                    det = {"class_name": reg["name"],
                           "confidence": reg["score"],
                           "bbox": reg["bbox"],
                           "clip_hint": (reg["name"], reg["category"],
                                         reg["default_shelf_life_days"])}
                    crop_texts = ocr.texts_inside(pil, reg["bbox"],
                                                  pad=config.CROP_PAD,
                                                  items=full_ocr)
                    d = _build_draft(
                        db, det, " ".join(crop_texts), crop_texts, today,
                        all_texts=match_texts, clip_names=clip_names)
                    drafts.append(_mark_region_draft(db, d, reg["bbox"],
                                                     i, len(split),
                                                     width, height))
                region_note = (
                    f"画面上摆着多件物品（整图视觉判类只有 {c['score']:.0%}），"
                    f"已按视觉特征切分出 {len(drafts)} 个物品区域，"
                    f"每格里的名称与日期请逐个确认")
            else:
                det = {"class_name": c["name"],
                       "confidence": c["score"],
                       "bbox": [0, 0, width, height],
                       "clip_hint": (c["name"], c["category"],
                                     c["default_shelf_life_days"])}
                drafts.append(_build_draft(
                    db, det, " ".join(match_texts), match_texts, today,
                    all_texts=match_texts, clip_names=clip_names))
                region_note = (f"包装上无文字，按视觉特征识别为「{c['name']}」，"
                               f"请确认")
        else:
            # 5.2 区域 CLIP 多物品扫描：YOLO 对芝士盒/黄油/保鲜膜包装等
            #     非 COCO 物品整类漏检、OCR 又读不到文字时（冰箱/货架
            #     远摄场景），按多尺度网格逐区域零样本分类，识别多个物品。
            regions = region_cls.scan_regions(pil) if not is_screenshot else []
            region_drafts: List[DetectionDraft] = []
            for reg in regions:
                det = {"class_name": reg["name"],
                       "confidence": reg["score"],
                       "bbox": reg["bbox"],
                       "clip_hint": (reg["name"], reg["category"],
                                     reg["default_shelf_life_days"])}
                crop_texts = ocr.texts_inside(pil, reg["bbox"],
                                              pad=config.CROP_PAD,
                                              items=full_ocr)
                date_text = " ".join(crop_texts)
                region_drafts.append(_build_draft(
                    db, det, date_text, crop_texts, today,
                    all_texts=match_texts, clip_names=clip_names))
            if region_drafts:
                drafts = region_drafts
                region_note = (f"包装上看不到可读文字，已按视觉特征识别出 "
                               f"{len(region_drafts)} 个物品区域，请逐个确认")
        if not drafts:
            # 连日期文字都没有：尝试按包装关键词 / CLIP 视觉给一条带名称建议的草稿
            hint = _ocr_hint(match_texts)
            if hint:
                h_name, h_cat_name, h_days = hint
                cat = _category_by_name(db, h_cat_name) or _category_by_name(db, "其他")
                drafts.append(DetectionDraft(
                    name=h_name, raw_name="", confidence=None, bbox=[],
                    category_id=cat.id if cat else 0,
                    category_name=cat.name if cat else "其他",
                    needs_review=_decide_review(None, None, _get_low_conf(db)),
                    candidates=_candidates_for(db, cat, raw_name=h_name,
                                               texts=match_texts, clip_names=clip_names),
                    date_note="未找到日期文字，请手动填写生产日期/保质期",
                ))
            elif clip_top and clip_top[0]["score"] >= 0.50:
                c = clip_top[0]
                cat = _category_by_name(db, c["category"]) or _category_by_name(db, "其他")
                drafts.append(DetectionDraft(
                    name=c["name"], raw_name="", confidence=c["score"], bbox=[],
                    category_id=cat.id if cat else 0,
                    category_name=cat.name if cat else "其他",
                    needs_review=_decide_review(c["score"], None, _get_low_conf(db)),
                    candidates=_candidates_for(db, cat, raw_name=c["name"],
                                               clip_names=clip_names),
                    date_note=(f"包装上未读到文字，视觉模型（CLIP）初判为「{c['name']}」"
                               f"（{c['score']:.0%}），请确认并填写日期"),
                ))
            else:
                drafts.append(DetectionDraft(
                    name="", raw_name="", confidence=None, bbox=[],
                    category_id=_default_category_id(db, "其他"),
                    category_name="其他",
                    needs_review=_decide_review(None, None, _get_low_conf(db)),
                    candidates=_candidates_for(db, _category_by_name(db, "其他"),
                                               texts=match_texts, clip_names=clip_names),
                    date_note="未识别到日期文字，请从候选中选择或手动填写",
                ))

    # 6. 清洗合并：去噪声聚类、并日期草稿、补名称、补保质期天数
    drafts = _consolidate_drafts(db, drafts, match_texts, clip_top, is_screenshot)

    # 7. 结果消息（按合并后的真实情况生成）
    message = ""
    if is_screenshot:
        message = "检测到上传的是界面截图，已忽略界面文字、按截图中的照片内容识别。"
    if supplement_note:
        message += supplement_note
    if fallback_used:
        if region_note:
            message += region_note
        elif all(d.name for d in drafts):
            names = "、".join(dict.fromkeys(d.name for d in drafts if d.name))
            message += (f"通用检测模型未识别出该物品，已改用包装文字定位，"
                        f"识别出 {len(drafts)} 个物品区域（{names}）；"
                        f"名称与日期均由包装文字读取，建议核对日期")
        else:
            message += (f"通用检测模型未识别出该物品，已按包装文字的日期/名称区域"
                        f"生成 {len(drafts)} 条草稿，请补全物品名称")

    return {
        "filename": image_path.name,
        "image_url": f"/uploads/{rel_path}",
        "image_width": width,
        "image_height": height,
        "ocr_texts": full_texts,
        "drafts": drafts,
        "message": message,
        "clip_top": clip_top,
        "is_screenshot": is_screenshot,
    }


# ---------------------------------------------------------------- OCR 区域聚类

_DATE_KEYWORD = re.compile(
    r"(生产日期|保质期|有效期|最佳食用期|保鲜期|BEST\s*BEFORE|USE\s*BY|EXP\b|EXPIRY|MFG|PROD|\bPD\b|\bED\b)",
    re.I)
_RE_ANCHOR_DATE = re.compile(
    r"(?<!\d)(?:\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2})?日?|20\d{6}|20\d{2}\s+(?:0[1-9]|1[0-2])\s+(?:0[1-9]|[12]\d|3[01]))(?!\d)")


def _is_date_anchor(text: str) -> bool:
    """该行是否作为"日期区域"锚点。

    只有两种：① 含日期关键词（生产日期/保质期至/Prod/Exp...）；
    ② 含可解析的日期值（2021.09.02、2024年3月、20240315）。
    条码/电话/批号（64802223、029356000047）不是合法年月日，不当锚点——
    旧规则把 8 位数字当锚点，导致从条码区生成大量垃圾草稿。
    """
    t = normalize_text(text)
    if not t:
        return False
    if _DATE_KEYWORD.search(t):
        return True
    for m in _RE_ANCHOR_DATE.finditer(t):
        if _parse_date_token(m.group(0)):
            return True
    return False


def _is_name_anchor(text: str) -> bool:
    """该行是否携带"产品名证据"（名称聚类模式的锚点）。

    两类：① 命中包装关键词（浆果/椰子水/安格斯...）；
    ② 含词典物品名全名（"酸奶"、"鸡蛋 30枚"命中"鸡蛋"）。
    注意词典部分匹配（"曲奇/饼干"拆出的"饼干"）不当锚点——烹饪说明
    "加入适量面条"会命中"挂面/面条"的"面条"部分，生成垃圾草稿
    （实测意大利面照片误判）；部分匹配只用于候选排序，不用于锚定。
    纯乱码/标签词/英文标语不当锚点——实测购物车照片的
    "PRODUCECF BDOTRAINCA"（PRODUCED BY...）若当锚点会生成垃圾草稿。
    """
    t = normalize_text(text)
    if not t:
        return False
    if _ocr_hint([t]):
        return True
    return any(name in t for name, _, _ in config.CANDIDATE_DICTIONARY)


def _bbox_of(box) -> tuple:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (min(xs), min(ys), max(xs), max(ys))


# 点阵通道提示框：短行 + 含日期关键词或 6-8 位日期样数字串。
# 点阵喷码行 DBNet 检测不到，但框的位置通常压在喷码行上，窄条扫描
# 这些区域比全图墨迹带扫描快得多（曲奇实测：提示区 2 条窄条即读出
# MFG/EXP 两行日期，全图带扫描要 80+ 次读取）。
_RE_HINT_DIGITS = re.compile(r"(?<!\d)\d{6,8}(?!\d)")


def _date_hint_boxes(ocr_items: List[Dict], limit: int = 6) -> List[tuple]:
    """从 OCR 检测框提取疑似日期区提示框，按可能性排序。

    ① 含日期关键词（生产日期/MFG/EXP...）最可能是喷码行所在；
    ② 含 6-8 位日期样数字串次之；短行、数字占比高的行更优先
    （地址/电话等长混排行排在后面，扫不到日期也不浪费多少预算）。
    """
    scored: List[tuple] = []
    for t in ocr_items:
        s = normalize_text(t.get("text", ""))
        if not s or len(s) > 24:
            continue
        kw = bool(_DATE_KEYWORD.search(s))
        dg = _RE_HINT_DIGITS.search(s)
        if not (kw or dg):
            continue
        box = t.get("box")
        if not box:
            continue
        bb = _bbox_of(box)
        if bb[2] - bb[0] < 8 or bb[3] - bb[1] < 8:
            continue
        digit_ratio = sum(c.isdigit() for c in s) / max(1, len(s))
        score = (100 if kw else 0)
        score += (40 if dg and len(dg.group(0)) >= 8 else 20 if dg else 0)
        score += 20 if digit_ratio > 0.6 else 0
        score -= len(s) / 2.0
        # 去掉被已有高分框基本包含的重复框
        if any(b[0] <= bb[0] + 8 and b[1] <= bb[1] + 8 and
               b[2] >= bb[2] - 8 and b[3] >= bb[3] - 8 for _, b in scored):
            continue
        scored.append((score, bb))
    scored.sort(key=lambda x: -x[0])
    return [bb for _, bb in scored[:limit]]


def _cy(box) -> float:
    return sum(p[1] for p in box) / 4


def _union_bbox(a: List[int], b: List[int]) -> List[int]:
    if not a:
        return [int(v) for v in b]
    if not b:
        return [int(v) for v in a]
    return [int(min(a[0], b[0])), int(min(a[1], b[1])),
            int(max(a[2], b[2])), int(max(a[3], b[3]))]


def _iou_bbox(a, b) -> float:
    ox = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ox * oy
    if inter <= 0:
        return 0.0
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (sa + sb - inter) if sa + sb - inter > 0 else 0.0


def _contained_ratio(small, big) -> float:
    """small 被 big 包含的面积比例（0-1）。"""
    if not small or not big:
        return 0.0
    ox = max(0, min(small[2], big[2]) - max(small[0], big[0]))
    oy = max(0, min(small[3], big[3]) - max(small[1], big[1]))
    inter = ox * oy
    sa = (small[2] - small[0]) * (small[3] - small[1])
    return inter / sa if sa > 0 else 0.0


def _adjacent_bbox(a, b, gap: int = 20) -> bool:
    """两个框是否在某方向上紧贴（间隙<gap 且垂直/水平重叠>60%）。

    YOLO 常把一个物体拆成左右/上下紧贴的两个框（实测苹果照片
    3 个苹果被检出 4 个框，其中两个仅 x 相接），合并它们避免
    同物多条草稿。
    """
    if not a or not b:
        return False
    gap_x = max(a[0] - b[2], b[0] - a[2])
    gap_y = max(a[1] - b[3], b[1] - a[3])
    h_min = min(a[3] - a[1], b[3] - b[1])
    w_min = min(a[2] - a[0], b[2] - b[0])
    if gap_x < gap and h_min > 0 and \
            (min(a[3], b[3]) - max(a[1], b[1])) / h_min > 0.6:
        return True
    if gap_y < gap and w_min > 0 and \
            (min(a[2], b[2]) - max(a[0], b[0])) / w_min > 0.6:
        return True
    return False


# 通用容器/杂物类：COCO 里非食品的日常物件。这些类在复杂背景里误报率
# 极高（实测苹果照片把背景瓶状物检出 4 个 bottle，conf 0.61-0.73；薯片
# 照片误检 book）。判定规则见主流程第 4 步：低置信且框内读不到文字的
# 才算背景误报（黄油盒虽被认成 bottle，但框内印着"黄油"——文字证据可用）。
_GENERIC_CLASSES = {
    "bottle", "bowl", "cup", "book", "vase", "potted plant", "clock",
    "handbag", "backpack", "suitcase", "wine glass", "fork", "knife",
    "spoon", "cell phone", "remote", "keyboard", "mouse",
}
_GENERIC_CONF = 0.75


# COCO 生鲜类（apple/orange/banana/西兰花/胡萝卜）：类内混淆严重
# （实测橙子被认成 apple），低置信框的名称以整图 CLIP 细分为准。
_FRESH_COCO = {"apple", "orange", "banana", "broccoli", "carrot"}
_FRESH_CONF = 0.75

# 通用检测最易把零食袋/包装误检成的类：薯片袋形似书(book)、方盒形似瓶/碗。
# 命中且拍的是条码密集的包装背面、又读不到品名时，不硬给"书籍本册"这种
# 错误名称，改为留空、让用户从候选选择（实测薯片背面被 YOLO 认成 book）。
_MISLEAD_CLASSES = {"book", "bottle", "cup", "bowl", "vase"}


def _clip_item_meta(name: str) -> Optional[tuple]:
    """从 CLIP 提示词表反查物品名 -> (类别名, 默认天数)。"""
    for _, n, cname, days in clip_cls.PROMPT_ITEMS:
        if n == name:
            return (cname, days)
    return None


def _drafts_from_ocr(db: Session, full_ocr: List[Dict],
                     today: dt.date, is_screenshot: bool = False,
                     name_only: bool = False,
                     skip_boxes: Optional[List[tuple]] = None) -> List[DetectionDraft]:
    """
    检测不到目标（或检出不足）时：把 OCR 文本行按空间邻近聚类，
    每个聚类（包装区域）生成一条草稿，并向上寻找相邻的商品名称行。

    name_only=True 时只按"产品名证据行"（命中包装关键词或词典物品名）
    聚类，用于 YOLO 只检出部分物品的购物车/货架照片补漏——莓果袋、
    榴莲蛋糕、蔬菜包等非 COCO 包装没有检测框，但包装文字读得到
    （实测购物车照片：2 个瓶子之外补出浆果/水果/生鲜肉类/蛋糕 4 条）。
    skip_boxes：已有草稿的 bbox，被覆盖的聚类跳过，避免同一包装两条草稿。
    """
    items = _filter_ui_lines(full_ocr, is_screenshot)
    if not items:
        return []
    lines = []
    for t in items:
        bb = _bbox_of(t["box"])
        anchor = _is_name_anchor(t["text"]) if name_only else _is_date_anchor(t["text"])
        lines.append({"text": t["text"], "bbox": bb, "cy": _cy(t["box"]),
                      "is_date": anchor})
    lines.sort(key=lambda l: (l["cy"], l["bbox"][0]))

    # 1. 锚点分组：y 相近且 x 有重叠
    groups: List[dict] = []
    for l in [x for x in lines if x["is_date"]]:
        for g in groups:
            if abs(l["cy"] - g["cy_avg"]) < 90 and \
               min(g["bbox"][2], l["bbox"][2]) - max(g["bbox"][0], l["bbox"][0]) > -10:
                g["lines"].append(l)
                g["bbox"] = (min(g["bbox"][0], l["bbox"][0]), min(g["bbox"][1], l["bbox"][1]),
                             max(g["bbox"][2], l["bbox"][2]), max(g["bbox"][3], l["bbox"][3]))
                g["cy_avg"] = sum(x["cy"] for x in g["lines"]) / len(g["lines"])
                break
        else:
            groups.append({"lines": [l], "bbox": l["bbox"], "cy_avg": l["cy"]})

    # 2. 垂直相邻且 x 重叠的组视为同一区块（如 "生产日期及保质期至：" 标签行
    #    与 "Prod: 2021.09.02" 数值行分开识别时）。
    #    名称聚类模式用更小的间距：购物车/货架照片物品堆叠密集，相邻包装
    #    的文字只差 ~80px（实测莓果袋文字与牛肉托文字 84px），150px 会把
    #    两个不同物品并成一条草稿；名称不需要跨行拼日期，70px 够用。
    merge_gap = 70 if name_only else 150
    merged: List[dict] = []
    for g in groups:
        target = None
        for m in merged:
            gap = max(g["bbox"][1] - m["bbox"][3], m["bbox"][1] - g["bbox"][3])
            x_overlap = min(g["bbox"][2], m["bbox"][2]) - max(g["bbox"][0], m["bbox"][0])
            if gap < merge_gap and x_overlap > -10:
                target = m
                break
        if target:
            target["lines"].extend(g["lines"])
            tb, gb = target["bbox"], g["bbox"]
            target["bbox"] = (min(tb[0], gb[0]), min(tb[1], gb[1]),
                              max(tb[2], gb[2]), max(tb[3], gb[3]))
            target["cy_avg"] = sum(x["cy"] for x in target["lines"]) / len(target["lines"])
        else:
            merged.append(g)

    # 3. 每组向上找商品名称行（260px 内、x 有重叠的非锚点行）
    # 单件包装判定：只有 ≤1 组带日期时，日期区与品名区常相距很远
    # （实测进口黄油点阵日期在包装顶部、意大利面日期区上方只有配料行），
    # 允许全文关键词补名；多物品场景（>1 组带日期）只用本地证据，
    # 防止别件物品的文字污染（实测模拟货架"鸡蛋"被带成"酸奶"）。
    n_dated = sum(
        1 for g in merged[:12]
        if parse_date_text(" ".join(l["text"] for l in g["lines"])).has_any)
    single_pack = n_dated <= 1
    drafts: List[DetectionDraft] = []
    for g in merged[:12]:
        if name_only and skip_boxes and \
                _covered_by_boxes(g["bbox"], skip_boxes):
            continue                     # 已被 YOLO/区域 CLIP 草稿覆盖
        top = g["bbox"][1]
        if name_only:
            # 名称聚类：锚点行本身就是产品名，不做 260px 向上名称搜索——
            # 会把上方相邻包装的文字并进来（实测牛肉托草稿的 bbox
            # 一路跨到莓果袋，高亮框错位）。只附带宽 70px 内的非锚点行
            # （产品名下方常印日期值）。
            gx1, gy1, gx2, gy2 = g["bbox"]
            titles = [l for l in lines if not l["is_date"]
                      and gy1 - 70 <= l["cy"] <= gy2 + 70
                      and min(gx2, l["bbox"][2]) - max(gx1, l["bbox"][0]) > -10]
        else:
            titles = [l for l in lines if not l["is_date"]
                      and top - 260 <= l["cy"] < top
                      and min(g["bbox"][2], l["bbox"][2]) - max(g["bbox"][0], l["bbox"][0]) > -10]
        titles.sort(key=lambda l: l["cy"])
        all_lines = sorted(titles + g["lines"], key=lambda l: l["cy"])
        texts = [l["text"] for l in all_lines]
        date_text = " ".join(texts)
        parsed = parse_date_text(date_text)

        # 名称建议：包装关键词（名称聚类模式下优先，且锚点行证据最强）
        # -> 词典匹配（全名最长优先，支持 "/" 部分匹配）
        suggested_name = ""
        matched = None  # 防止后续 if matched else 在分支未赋值时报 UnboundLocalError
        if name_only:
            hint = _ocr_hint([l["text"] for l in g["lines"]]) or _ocr_hint(texts)
            if hint:
                matched = hint
                # 产品名行含类名组成部分 -> 用具体产品名（"混合莓果"含"莓果"）
                prod_name = _clean_product_name(
                    _product_name_from_texts(_name_evidence_texts(texts)))
                suggested_name = _specific_name(prod_name, hint) or hint[0]
        if not suggested_name:
            hint = _ocr_hint(texts)
            if not hint and (not parsed.has_any or single_pack):
                # 读不到日期值的聚类（标签行/条码区，通常是同一件包装的
                # 噪声）或单件包装：允许全文关键词。多物品场景的带日期
                # 聚类只用本地证据——别件物品的文字会污染本草稿
                # （实测模拟货架"鸡蛋 30枚"被酸奶盒文字带成"酸奶"）。
                hint = _ocr_hint([t["text"] for t in items])
            # 食品名称标注 > 具体产品名（标题行）> 关键词类名 > 词典
            # 产品名行走 _name_evidence_texts：剔除配料表区块，配料里的
            # "植物油"不会再把蛋黄桃酥带成食用油（实测）
            # "食品名称：XXX"标注是最强证据，若聚类窗口没覆盖到它，回退整图
            # OCR 全文——实测红豆奶茶：标注在包装另一处，聚类里只剩营养成分
            # 表，名称被误取成"一反式脂肪"。
            label_name = _food_label_name(texts) or _food_label_name([t["text"] for t in items])
            prod_name = _clean_product_name(
                _product_name_from_texts(_name_evidence_texts(texts)))
            if hint:
                matched = hint
                suggested_name = label_name or _specific_name(prod_name, hint) or hint[0]
            elif label_name:
                suggested_name = label_name
            elif prod_name:
                suggested_name = prod_name
        if not suggested_name:
            matched = _name_from_texts(texts)
            if matched:
                parts = set([matched[0]] + _name_parts(matched[0]))
                for t in texts:
                    if any(p in t for p in parts):
                        suggested_name = t if (len(t) <= 8 and not any(c.isdigit() for c in t)) else matched[0]
                        break
        cat = _category_by_name(db, matched[1]) if matched else _category_by_name(db, "其他")
        if cat is None:
            cat = db.query(Category).first()
        default_days = matched[2] if matched else (cat.default_shelf_life_days if cat else 365)

        prod, days, expiry, notes = suggest_expiry(
            parsed.production_date, parsed.shelf_life_days, default_days,
            expiry_date=parsed.expiry_date, today=today)
        if parsed.warnings:
            notes.extend(parsed.warnings)
        if name_only and suggested_name:
            # 名称聚类草稿来自文字而非轮廓，明确告知用户确认
            notes = [f"按包装文字识别为「{suggested_name}」，未检测到该物品轮廓"] + notes
        notes = _refine_date_notes(notes, parsed, [t["text"] for t in items])

        bbox = [int(v) for v in all_lines[0]["bbox"]]
        for l in all_lines[1:]:
            b = l["bbox"]
            bbox = [int(min(bbox[0], b[0])), int(min(bbox[1], b[1])),
                    int(max(bbox[2], b[2])), int(max(bbox[3], b[3]))]
        drafts.append(DetectionDraft(
            name=suggested_name,
            raw_name="",
            confidence=None,
            bbox=bbox,
            category_id=cat.id if cat else 0,
            category_name=cat.name if cat else "其他",
            quantity=1,
            unit="件",
            production_date=prod,
            shelf_life_days=days,
            expiry_date=expiry,
            date_text=date_text[:200],
            date_source="ocr" if parsed.has_any else "manual",
            date_note="；".join(notes) or "请从候选中选择物品名称",
            needs_review=_decide_review(None, parsed, _get_low_conf(db)),
            candidates=_candidates_for(db, cat, raw_name=suggested_name,
                                       texts=texts),
        ))

    # 单件包装的日期区与品名区可能相距很远（实测进口黄油：点阵喷码在
    # 包装顶部，聚类内除日期行外没有文字，名称一直空着）。只有一条草稿
    # 带日期时（单件包装），允许用全文关键词/产品名补它的名称；多物品
    # 场景禁用，避免别件物品的文字污染（实测模拟货架"鸡蛋"被带成"酸奶"）。
    dated = [d for d in drafts if d.production_date or d.expiry_date]
    if len(dated) == 1:
        if dated[0].production_date and dated[0].expiry_date:
            # 生产+到期都读到了：保质期天数以两者差值为准（关键词默认
            # 天数只是读不到时长时的兜底——实测黄油被默认天数填成 7 天）
            dated[0].shelf_life_days = (
                dated[0].expiry_date - dated[0].production_date).days
        if not dated[0].name or (dated[0].name and _is_class_name(dated[0].name)):
            full_texts = [t["text"] for t in items]
            hint = _ocr_hint(full_texts)
            label_name = _food_label_name(full_texts)
            prod_name = _clean_product_name(
                _product_name_from_texts(_name_evidence_texts(full_texts)))
            name = label_name or _specific_name(prod_name, hint) or \
                (hint[0] if hint else None) or prod_name
            # 名称升级：原名称只是关键词类名（"辣酱/酱类"），而包装上有更
            # 具体的产品名（"拌饭辣酱"）时，换成产品名
            if name and name != dated[0].name:
                dated[0].name = name
                if hint:
                    h_cat = _category_by_name(db, hint[1])
                    if h_cat:
                        dated[0].category_id = h_cat.id
                        dated[0].category_name = h_cat.name
                        if not dated[0].shelf_life_days:
                            dated[0].shelf_life_days = hint[2]
                dated[0].date_note = _append_note(
                    dated[0].date_note, f"包装文字读作「{name}」（名称来自包装文字）")
    return drafts


# ---------------------------------------------------------------- 草稿清洗合并

def _name_from_texts(texts: List[str]) -> Optional[tuple]:
    """全文词典匹配：全名命中取最长；否则按命中部分数取部分匹配。

    返回 (名称, 类别, 默认天数) 或 None。
    """
    joined = normalize_text(" ".join(texts or []))
    if not joined:
        return None
    full: Optional[tuple] = None
    partial: Optional[tuple] = None
    partial_hits = 0
    for name, cname, days in config.CANDIDATE_DICTIONARY:
        mp = _matched_parts(name, joined)
        if not mp:
            continue
        if name in mp:
            if full is None or len(name) > len(full[0]):
                full = (name, cname, days)
        elif len(mp) > partial_hits:
            partial, partial_hits = (name, cname, days), len(mp)
    return full or partial


def _append_note(note: str, extra: str) -> str:
    if not note:
        return extra
    if extra in note:
        return note
    return note + "；" + extra


def _consolidate_drafts(db: Session, drafts: List[DetectionDraft],
                        name_texts: List[str], clip_top: List[Dict],
                        is_screenshot: bool) -> List[DetectionDraft]:
    """
    草稿清洗与合并（识别质量的最后一道关）：
    1. 日期值相同的草稿合并（同一包装的 Prod/Exp 被拆成多个聚类）；
    2. 存在"有日期"草稿时，丢弃无日期且无名称的草稿
       （条码/配料表/营养成分表等噪声聚类）；
    3. 无名称草稿补全名称：全文词典匹配 > CLIP 视觉初判；
    4. 生产日期+到期日期齐全时补算保质期天数；
    5. 重排序：有日期、有名称的草稿在前。

    name_texts 已按截图场景过滤掉界面文字（候选按钮/表单标签/顶栏日期），
    避免把界面 chip 当成物品名、把顶栏日期当成生产日期。
    """
    if not drafts:
        return drafts
    dated = [d for d in drafts if d.production_date or d.expiry_date]

    # 1) 相同日期集合并
    merged: List[DetectionDraft] = []
    for d in dated:
        key = (d.production_date, d.expiry_date)
        hit = next((m for m in merged
                    if (m.production_date, m.expiry_date) == key), None)
        if hit:
            hit.date_text = (hit.date_text + " " + d.date_text).strip()[:200]
            hit.bbox = _union_bbox(hit.bbox, d.bbox)
            if not hit.name and d.name:
                hit.name = d.name
            if d.confidence and (hit.confidence is None or d.confidence > hit.confidence):
                hit.confidence = d.confidence
            if d.candidates and not hit.candidates:
                hit.candidates = d.candidates
        else:
            merged.append(d)

    others = [d for d in drafts if not (d.production_date or d.expiry_date)]
    named = [d for d in others if d.name]
    # 同类空间重叠合并（无日期草稿）：YOLO 对同一物品重复出框（实测
    # 苹果照片 3 个 apple 框 -> 3 条"苹果"草稿），同名且 IoU>0.5 或
    # 小框基本落在大框内（>0.8）时合并取并集、保留最高置信度。
    # 例外：多物品切分区域的草稿是"一格一件"的独立物品，冰箱里并排摆着
    # 三包同款芝士时，按相邻规则合并会把三件并成一件——这类草稿只合并
    # 几乎完全重合的框（确实是同一块区域被重复生成）。
    if len(named) > 1:
        collapsed: List[DetectionDraft] = []
        for d in sorted(named, key=lambda x: -(x.confidence or 0)):
            hit = None
            for m in collapsed:
                region_pair = d.region_split or m.region_split
                if m.name != d.name:
                    continue
                if region_pair:
                    same_spot = _iou_bbox(d.bbox, m.bbox) > 0.7
                else:
                    same_spot = (_iou_bbox(d.bbox, m.bbox) > 0.5
                                 or _contained_ratio(d.bbox, m.bbox) > 0.8
                                 or _adjacent_bbox(d.bbox, m.bbox))
                if same_spot:
                    hit = m
                    break
            if hit:
                hit.bbox = _union_bbox(hit.bbox, d.bbox)
                hit.quantity = (hit.quantity or 1) + (d.quantity or 1)
                if d.confidence and (hit.confidence is None
                                     or d.confidence > hit.confidence):
                    hit.confidence = d.confidence
                if d.candidates and not hit.candidates:
                    hit.candidates = d.candidates
            else:
                collapsed.append(d)
        named = collapsed
    # 唯一"有日期"草稿存在时，同名无日期草稿是同一包装被聚类拆开的残留，丢弃
    # （多物品切分区域例外：它们是画面里不同的实物，同款也得各留一条）
    if len(merged) == 1:
        mname = merged[0].name
        # 无日期草稿的名称若是主品名的子串（"奶茶" ⊂ "红豆奶茶"），多半是同一
        # 包装被文字补漏/聚类拆出的总称或配料碎片，丢弃避免同一物品出多张卡
        # （实测红豆奶茶：主卡"红豆奶茶"+碎片卡"奶茶"）。
        named = [d for d in named
                 if d.region_split or not (mname and d.name
                                           and (d.name == mname or d.name in mname))]
    # 无名称的草稿（条码/配料表聚类、待补名称）一律保留；有名称的用合并后的
    # 结果——原先这里写 `named if merged else others`，等于"没有任何日期草稿
    # 时合并白做"，同款相邻的两条草稿会同时留下（实测两张并排的酸奶各出
    # 一条卡片，第一张的框还被并成了两件的并集）。
    kept = merged + named + [d for d in others if not d.name]

    # 2) 名称补全：全文词典 > CLIP 视觉初判
    for d in kept:
        if d.name:
            continue
        hit = _name_from_texts(name_texts)
        if hit:
            d.name = hit[0]
            if not d.category_id or d.category_name == "其他":
                cat = _category_by_name(db, hit[1])
                if cat:
                    d.category_id, d.category_name = cat.id, cat.name
            d.date_note = _append_note(d.date_note, f"物品名称取自包装文字「{hit[0]}」")
        elif clip_top and clip_top[0]["score"] >= 0.50:
            c = clip_top[0]
            d.name = c["name"]
            cat = _category_by_name(db, c["category"])
            if cat:
                d.category_id, d.category_name = cat.id, cat.name
            d.date_note = _append_note(
                d.date_note,
                f"包装文字未识别到物品名，视觉模型（CLIP）初判为「{c['name']}」"
                f"（{c['score']:.0%}）")

    # 3) 保质期天数补算（生产+到期都有但天数为空）
    for d in kept:
        if d.production_date and d.expiry_date and not d.shelf_life_days:
            d.shelf_life_days = (d.expiry_date - d.production_date).days

    # 4) 重排序：有日期在前，有名称在前
    kept.sort(key=lambda d: (0 if (d.production_date or d.expiry_date) else 1,
                             0 if d.name else 1))
    return kept


def _default_category_id(db: Session, name: str) -> int:
    c = _category_by_name(db, name)
    if c:
        return c.id
    c = db.query(Category).first()
    return c.id if c else 1


def _refine_date_notes(notes: List[str], parsed, all_texts: List[str]) -> List[str]:
    """无任何日期时：若全量 OCR 文本里有"日期见背面"类精准提示，替换通用兜底文案。"""
    if parsed.has_any or not all_texts:
        return notes
    hint = date_elsewhere_hint(" ".join(all_texts))
    if not hint:
        return notes
    notes = [n for n in notes
             if "可能印在包装其他位置" not in n and "未识别到日期信息" not in n]
    if hint not in notes:
        notes.append(hint)
    return notes


def _get_low_conf(db: Session) -> float:
    """从设置表读取低置信度审核阈值；未配置/异常时回退到 config.LOW_CONF。"""
    try:
        s = db.get(Setting, 1)
        if s is not None and s.low_conf is not None:
            return float(s.low_conf)
    except Exception:
        pass
    return float(config.LOW_CONF)


# 生产日期早于"今天 - N 年"视为明显可疑（疑似把旧货/过期喷码当新货入库）。
# 仅对这类日期明显不可信的物品强制进待确认，避免把 OCR 读到的正常日期也退回。
SUSPECT_OLD_YEARS = 5


def _decide_review(conf, parsed, low_conf: float) -> bool:
    """统一的"是否需要人工确认"判定。

    仅当存在"硬"不确定信号时才进入待确认，让高置信且读到期日的物品
    拍照即自动入库，系统才真正省事：
      - 置信度过低（conf 非空且 < low_conf）：识别把握不足；
      - 未识别到任何日期：到期日只能按类别默认估算，不可靠；
      - 日期解析有告警（parsed.warnings 非空）：日期歧义，可能读错；
      - 日期明显可疑：算出来已过期，或生产日期比今天早 N 年以上
        （疑似把旧货/错读年份入库，需要人眼核对）。

    注意：AI 补全名称所产生的 hint_notes 仅作为"推理说明"展示，
    不再作为阻塞自动确认的条件（之前几乎每件都因"请确认"被卡住）。
    """
    low_conf_flag = (conf is not None) and (conf < low_conf)
    no_date = True if parsed is None else (not parsed.has_any)
    warnings = bool(getattr(parsed, "warnings", None)) if parsed else False
    date_suspect = False
    if parsed is not None:
        today = dt.date.today()
        prod = getattr(parsed, "production_date", None)
        exp = getattr(parsed, "expiry_date", None)
        if prod is not None and prod.year <= today.year - SUSPECT_OLD_YEARS:
            date_suspect = True
        if exp is not None and exp < today:
            date_suspect = True
    return bool(low_conf_flag or no_date or warnings or date_suspect)


def _build_draft(db: Session, det: Dict, date_text: str, texts: List[str],
                 today: dt.date, all_texts: Optional[List[str]] = None,
                 clip_names: Optional[List[str]] = None) -> DetectionDraft:
    """把一条检测结果加工成可审核的物品草稿。"""
    raw_name = det["class_name"]
    conf = det["confidence"]
    hint_notes: List[str] = []

    # 归类：包装文字证据 > 用户修正记忆 > COCO 静态映射 > 其他
    # 注意修正记忆的生效条件：它按"视觉类别"（raw_name）整体记忆，但同一
    # 视觉类别常对应不同产品——bottle 既可能是矿泉水也可能是椰子水/梨汁。
    # 实测购物车照片：软儿梨瓶与椰子水盒都被无条件盖成历史修正"矿泉水"，
    # 包装明明读到了产品名。因此区域文字给出矛盾的产品证据时，修正记忆
    # 降级为备注，让文字证据说话。
    correction = _correction_for(db, raw_name)
    corr_conflict = False
    if correction and correction.category_id and \
            not _correction_conflicts(correction.item_name or "", texts):
        cat = db.get(Category, correction.category_id)
        item_name = correction.item_name or (config.COCO_CLASS_MAP.get(raw_name, ("其他", raw_name, 365))[1])
    else:
        if correction and correction.category_id:
            corr_conflict = True
        mapped = config.COCO_CLASS_MAP.get(raw_name)
        cat = _category_by_name(db, mapped[0]) if mapped else None
        item_name = mapped[1] if mapped else raw_name
    if cat is None:
        cat = _category_by_name(db, "其他") or db.query(Category).first()
    default_days = cat.default_shelf_life_days if cat else 365

    # YOLO 生鲜类低置信框：COCO 的 apple/orange/banana 类内混淆严重
    # （实测橙子照片被认成 apple conf≈0.6），整图 CLIP 有细分结论时
    # 以 CLIP 为准（葡萄/橙子/梨/西瓜等 PROMPT_ITEMS 细分项）。
    if (not det.get("clip_hint") and raw_name in _FRESH_COCO
            and conf < _FRESH_CONF and clip_names and not correction):
        c = clip_names[0]
        meta = _clip_item_meta(c)
        if c and meta and c != item_name:
            item_name = c
            cat = _category_by_name(db, meta[0]) or cat
            default_days = meta[1]
            hint_notes.append(f"视觉模型（CLIP）初判为「{c}」")

    # 包装文字关键词证据：与 YOLO 视觉结论冲突时以文字为准（附说明并强制人工复核）
    if corr_conflict:
        hint_notes.append(
            f"包装文字与历史修正「{correction.item_name}」不符，已按包装文字识别")
    # 区域 CLIP 视觉分类（货架/冰箱多物品场景）：名称来自词典条目、
    # 自带类别与默认保质期，比 COCO 映射准；修正记忆生效时仍优先。
    if det.get("clip_hint") and not (correction and not corr_conflict):
        h_name, h_cat_name, h_days = det["clip_hint"]
        h_cat = _category_by_name(db, h_cat_name)
        if h_cat:
            cat, item_name, default_days = h_cat, h_name, h_days
            # 不要把 det 的视觉置信度当 CLIP 分数展示：二者不是一回事
            # （实测 0.37 是 YOLO "book" 的置信度，被错标成"CLIP 初判 37%"）。
            hint_notes.append(f"视觉模型（CLIP）初判为「{h_name}」")
    # 预包装"食品名称：XXX"标注最准确，优先于词典类名与 YOLO 类别名。
    # 区域切分场景（包装被 4x3 网格拆开）里该标注常落在别的区域、只在整图
    # OCR 里读得到：此时区域自身 texts 找不到，必须回退整图全文 all_texts。
    # 实测红豆奶茶：整图读到"食品名称：红豆奶茶"，区域里只剩营养成分表，
    # 名称遂被 _product_name_from_texts 误取成"一反式脂肪"。
    label_name = _food_label_name(texts) or _food_label_name(all_texts)
    hint = _ocr_hint(texts) or _ocr_hint(all_texts)
    if hint:
        h_name, h_cat_name, h_days = hint
        h_cat = _category_by_name(db, h_cat_name)
        if h_cat:
            if cat is None or h_cat.id != cat.id:
                cat, default_days = h_cat, h_days
                if not label_name:
                    item_name = h_name
                hint_notes.append(
                    f"包装文字显示为「{label_name or h_name}」，已按包装文字修正类别"
                    f"（视觉初判：{raw_name or '未知'}）")
            elif not label_name and item_name != h_name:
                item_name = h_name
                hint_notes.append(f"包装文字补充物品名称：「{h_name}」")
    if label_name:
        item_name = label_name
        hint_notes.append(f"包装食品名称标注为「{label_name}」")

    # 产品名行比关键词类名更具体时用产品名（"蛋黄桃酥"含"桃酥"，类名只定类别）
    if hint and not label_name:
        prod_name = _clean_product_name(
            _product_name_from_texts(_name_evidence_texts(texts)))
        specific = _specific_name(prod_name, hint)
        if specific and specific != item_name:
            item_name = specific
            hint_notes.append(f"包装文字读作「{specific}」，已按包装文字识别")

    # CLIP 视觉初判：包装无文字且 YOLO 结论为空时补充名称建议
    if not item_name and clip_names:
        c = next((x for x in clip_names if x), None)
        if c:
            item_name = c
            hint_notes.append(f"视觉模型（CLIP）初判为「{c}」")

    # 包装文字产品名兜底：关键词/食品名称标注/修正记忆/CLIP 都没能给出
    # 产品名时，取区域文字里第一条像产品名的行（实测购物车照片：
    # 软儿梨瓶读出「兰州软儿梨」，比 COCO 的"瓶装饮品"具体得多）。
    # 同样走 _name_evidence_texts：配料表明文（植物油/黄油）不当产品名。
    if not label_name and not hint and not det.get("clip_hint") and \
            not (correction and not corr_conflict):
        prod_name = _clean_product_name(
            _product_name_from_texts(_name_evidence_texts(texts)))
        if prod_name and prod_name != item_name:
            item_name = prod_name
            hint_notes.append(f"包装文字读作「{prod_name}」")

    # 通用检测对零食袋/包装的误判：book/bottle 等 + 条码密集的包装背面 +
    # 读不到品名时，不硬给"书籍本册/笔记本"这种错误确定答案，留空待选。
    # 其中 book 类（扁平包装最易被 YOLO 误判）只要没读到明确品名就直接进
    # 候选——储物架/冰箱场景里它几乎一定是零食袋而非真书；其余容器类
    # （bottle/cup...）额外要求条码密集（包装背面）才置空，避免把真瓶装
    # 饮品误清空。区域 CLIP 已高置信纠正的情况下也不走此兜底。
    mislead_no_name = (raw_name in _MISLEAD_CLASSES and not hint and not label_name
                       and not det.get("clip_hint"))
    if mislead_no_name and (raw_name == "book"
                            or _text_is_barcode_heavy(all_texts or texts)):
        item_name = ""
        hint_notes.append(
            f"通用检测可能误判（模型输出「{raw_name}」），未能可靠识别物品，"
            f"请从下方候选中选择")

    # 日期抽取与归一化
    parsed = parse_date_text(date_text)
    prod, days, expiry, notes = suggest_expiry(
        parsed.production_date, parsed.shelf_life_days, default_days,
        expiry_date=parsed.expiry_date, today=today)
    if prod and expiry and not days:
        days = (expiry - prod).days          # 生产+到期齐全时补算保质期天数
    date_source = "ocr" if parsed.has_any else "manual"
    if not parsed.has_any and default_days:
        # 未识别到任何日期：给一个"类别默认"的建议，需用户确认
        notes = notes or [f"未识别到日期，按类别默认 {default_days} 天估算"]
        date_source = "category_default"

    # 包装明文确认名称 → 置信度以文字为准（见 config.TEXT_EVIDENCE_CONF）。
    # label_name（"食品名称：XXX"标注）与 hint（品类关键词命中）都来自包装印刷，
    # 比 COCO 视觉可靠：YOLO 把黄油盒误判成 book(0.37)，但包装正印着
    # "产品名称：…黄油卷"。若仍沿用视觉低置信，会把名称清清楚楚的商品标成
    # 待确认并抛候选让用户重选（用户实测反馈）。此处同时清掉与之矛盾的
    # "视觉初判为 XX"噪声 note——文字已确证，视觉猜错不必再展示。
    if label_name or hint:
        conf = max(conf if conf is not None else 0.0, config.TEXT_EVIDENCE_CONF)
        hint_notes = [n for n in hint_notes if "初判为" not in n]

    low_conf = _get_low_conf(db)
    needs_review = _decide_review(conf, parsed, low_conf)
    if parsed.warnings:
        notes.extend(parsed.warnings)
    if hint_notes:
        notes = hint_notes + notes
    notes = _refine_date_notes(notes, parsed, all_texts or texts)

    candidates: List[Candidate] = []
    # 候选名称只在"名称尚未确证为包装完整品名"时给出：
    # - label_name（"食品名称/产品名称：XXX"标注）= 包装印的完整品名，已最具体，
    #   不必让用户从候选里重选印着的大字（用户实测反馈）。
    # - 仅 hint 命中（泛化类别名如"维生素/黄油"）或无名称时，名称仍可能不够具体
    #   （"维生素" vs "维生素C"），给候选让用户精确选择（用户实测反馈）。
    # 名称仍可手动改。
    if needs_review and not label_name:
        candidates = _candidates_for(db, cat, raw_name=raw_name,
                                     texts=all_texts or texts,
                                     clip_names=clip_names)

    return DetectionDraft(
        name=item_name or raw_name,
        raw_name=raw_name,
        confidence=conf,
        bbox=det["bbox"],
        category_id=cat.id if cat else 0,
        category_name=cat.name if cat else "其他",
        quantity=1,
        unit="件",
        production_date=prod,
        shelf_life_days=days,
        expiry_date=expiry,
        date_text=date_text[:200],
        date_source=date_source,
        date_note="；".join(notes),
        needs_review=needs_review,
        candidates=candidates,
    )


# ---------------------------------------------------------------- 多物品切分草稿

def _region_pos_label(bbox: List[int], width: int, height: int) -> str:
    """把区域框换算成"第几行第几列"，方便用户对照照片找到这块区域。

    切分用的是 4x3 互不重叠网格，直接按框的起点反推网格行列。
    """
    if not bbox or len(bbox) != 4 or width <= 0 or height <= 0:
        return ""
    row = min(3, max(1, int(bbox[1] / (height / 3)) + 1))
    col = min(4, max(1, int(bbox[0] / (width / 4)) + 1))
    return f"第{row}行第{col}列"


def _mark_region_draft(db: Session, d: DetectionDraft, bbox: List[int],
                       index: int, total: int, width: int, height: int
                       ) -> DetectionDraft:
    """标一条"多物品切分区域"草稿：位置说明 + 强制候选确认。

    切分区域的名称来自 CLIP 单格判类（一张冰箱照片里几件东西很可能都被
    判成"芝士/奶酪"），因此即便分数很高也必须让用户确认：给候选列表、
    置 needs_review，并在说明里写清这是第几块区域。
    """
    d.region_split = True
    d.needs_review = True
    pos = _region_pos_label(bbox, width, height)
    head = f"多物品切分区域 {index}/{total}" + (f"（{pos}）" if pos else "")
    d.date_note = f"{head}；{d.date_note}" if d.date_note else head
    if not d.candidates:
        cat = db.get(Category, d.category_id) if d.category_id else None
        d.candidates = _candidates_for(db, cat, raw_name=d.raw_name,
                                       clip_names=[d.name] if d.name else None)
    return d


# ---------------------------------------------------------------- 修正记忆

def remember_correction(db: Session, raw_name: str, category_id: int, item_name: str = ""):
    """用户修改类别/名称后写入修正记忆，下次同 raw_name 优先沿用。"""
    if not raw_name:
        return
    mem = _correction_for(db, raw_name)
    if mem:
        if mem.category_id != category_id or (item_name and mem.item_name != item_name):
            mem.category_id = category_id
            if item_name:
                mem.item_name = item_name
        mem.use_count += 1
        mem.last_used_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    else:
        db.add(CorrectionMemory(raw_name=raw_name, category_id=category_id,
                                item_name=item_name, use_count=1,
                                last_used_at=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)))
    db.commit()
