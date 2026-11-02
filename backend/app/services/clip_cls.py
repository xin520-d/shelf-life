"""CLIP 零样本视觉识别服务（第二条识别通道，与 YOLO/RapidOCR 互补）。

为什么需要它：
  - YOLO11n 只认识 COCO 80 类通用物体（bowl/book/bottle...），
    对"意大利面/洗发水/感冒药"等包装商品大量误检或检不出；
  - RapidOCR 依赖包装上有文字，照片糊/角度歪/无文字时失效；
  - CLIP 图文匹配给出"视觉初判"，与包装文字证据互相印证。

模型：openai/clip-vit-large-patch14（约 1.7GB，CPU 推理 1-2s）。
选型说明：ViT-B/32 在真实商品照片上细粒度区分不足（实测羊奶盒上
"鲜牛奶"与"普洱茶"余弦差仅 0.03，softmax 后噪声被放大成错误高置信），
ViT-L/14 参数量 4 倍，意大利面照片零样本 0.83 直接命中。
未下载时所有接口优雅降级返回 None，不影响主流程。
"""
import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import config

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_DIR = config.MODEL_DIR / "clip_l14"       # HF 快照缓存目录
TEXT_EMB_FILE = CLIP_DIR / "text_emb.pt"       # 文本嵌入缓存
PROMPTS_FILE = CLIP_DIR / "prompts.json"       # 生成缓存时的提示词快照（用于校验）

_lock = threading.Lock()
_model = None          # CLIPModel
_processor = None      # CLIPProcessor
_text_emb = None       # torch.Tensor [N, D]，已归一化
_download_state = {"state": "idle", "error": ""}   # idle|downloading|ready|failed

# ---------------------------------------------------------------- 提示词表
# (描述短语列表, 物品名, 类别, 默认保质期天数) —— 与 CANDIDATE_DICTIONARY 对应。
# CLIP 的 BPE 词表以英文为主，用英文提示词效果远好于中文。
# 每个物品给 2-3 个描述做"提示词集成"（取最高分）：实测单模板时
# "a pack of dried spaghetti pasta" 在意大利面照片上只有 0.24，
# 加上 "a package of pasta / pasta in a plastic bag" 后达 0.80。
PROMPT_ITEMS: List[Tuple[List[str], str, str, int]] = [
    (["a carton of fresh milk", "a bottle of milk", "milk"], "纯牛奶", "食品", 7),
    (["a cup of yogurt", "yogurt in a plastic cup"], "酸奶", "食品", 14),
    (["a few eggs", "eggs in a carton"], "鸡蛋", "食品", 30),
    (["a block of butter", "butter in a paper wrapper"], "黄油", "食品", 90),
    (["a piece of cheese", "cheese in a package"], "芝士/奶酪", "食品", 30),
    (["a string of ham sausages", "a pack of ham sausage"], "火腿/香肠", "食品", 45),
    (["a loaf of bread", "bread in a plastic bag"], "面包", "食品", 5),
    (["sliced toast bread", "a pack of toast bread"], "吐司", "食品", 5),
    (["a bottle of fruit juice", "a carton of juice"], "果汁", "食品", 365),
    (["a can of cola soft drink", "a bottle of soda"], "可乐/汽水", "食品", 365),
    (["a bottle of mineral water", "a case of bottled water"], "矿泉水", "食品", 730),
    (["a pack of green tea", "green tea leaves in a box"], "绿茶", "食品", 540),
    (["a pack of black tea", "black tea leaves in a box"], "红茶", "食品", 730),
    (["a pack of oolong tea", "oolong tea leaves in a box"], "乌龙茶", "食品", 730),
    (["a disc of compressed pu'er tea", "pu'er tea in a package"], "普洱茶", "食品", 1095),
    (["a pack of scented flower tea", "jasmine tea in a box"], "花茶", "食品", 540),
    (["a pack of white tea", "white tea leaves in a box"], "白茶", "食品", 730),
    (["a pack of longjing tea leaves", "longjing tea in a box"], "龙井", "食品", 365),
    (["a pack of tieguanyin tea", "tieguanyin tea in a box"], "铁观音", "食品", 730),
    (["a pack of tea bags", "tea bags in a box"], "茶叶/茶包", "食品", 730),
    (["a cup of milk tea with pearls", "a cup of bubble tea"], "奶茶", "食品", 180),
    (["a pack of instant noodles", "instant noodles in a plastic bag"], "方便面", "食品", 180),
    (["a bag of potato chips", "a packet of crispy potato chips",
      "a yellow bag of puffed snack food"], "薯片/膨化食品", "食品", 270),
    (["a pack of chocolate chip cookies", "a box of cream sandwich biscuits",
      "a package of cookie snacks", "a yellow pack of filled biscuits"], "曲奇/饼干", "食品", 365),
    (["a jar of cashew nuts", "a bag of mixed nuts and walnuts",
      "roasted nuts in a glass jar", "a package of shelled peanuts"], "坚果/炒货", "食品", 180),
    (["a bar of chocolate", "a box of assorted candies",
      "colorful wrapped candy"], "糖果/巧克力", "食品", 365),
    (["a bag of coffee beans", "a jar of instant coffee powder",
      "a cup of black coffee"], "咖啡", "食品", 540),
    (["a box of western pastries", "a pack of cake and pastry",
      "a plate of assorted pastries"], "饼干/糕点", "食品", 365),
    (["a can of canned food", "canned food in a tin"], "罐头", "食品", 730),
    (["a bag of rice", "rice in a woven bag"], "大米", "食品", 365),
    (["a bottle of cooking oil", "cooking oil in a plastic bottle"], "食用油", "食品", 540),
    (["a bottle of soy sauce", "soy sauce in a glass bottle"], "酱油/调味", "食品", 540),
    (["a bottle of ketchup", "tomato ketchup in a bottle"], "番茄酱", "食品", 365),
    (["a pack of frozen dumplings", "frozen dumplings in a plastic bag"], "冷冻水饺", "食品", 180),
    (["a cone of ice cream", "a tub of ice cream"], "冰淇淋", "食品", 365),
    (["raw fresh meat on a tray", "raw meat in a plastic tray"], "生鲜肉类", "食品", 3),
    (["fresh fruits", "fruits in a basket"], "水果", "食品", 7),
    (["fresh vegetables", "vegetables in a basket"], "蔬菜", "食品", 5),
    (["bananas", "a bunch of bananas"], "香蕉", "食品", 7),
    (["apples", "a few red apples"], "苹果", "食品", 14),
    # 水果细分：用户常直接拍散装水果（无包装文字），CLIP 需要能分辨品种
    (["grapes", "a bunch of purple grapes"], "葡萄", "食品", 7),
    (["oranges", "a few oranges on a table"], "橙子", "食品", 14),
    (["pears", "a few green pears"], "梨", "食品", 7),
    (["a watermelon", "a whole watermelon"], "西瓜", "食品", 14),
    (["strawberries", "a few strawberries"], "草莓", "食品", 3),
    (["peaches", "a few peaches"], "桃子", "食品", 5),
    (["mangoes", "a few mangoes"], "芒果", "食品", 7),
    (["kiwi fruits", "a few kiwis"], "猕猴桃", "食品", 10),
    (["a dragon fruit", "a cut dragon fruit"], "火龙果", "食品", 7),
    (["lemons", "a few lemons"], "柠檬", "食品", 14),
    (["a pomelo", "a large pomelo fruit"], "柚子", "食品", 30),
    (["blueberries", "a box of blueberries"], "蓝莓", "食品", 5),
    (["a cantaloupe melon", "a cut cantaloupe"], "哈密瓜", "食品", 10),
    (["a durian", "a spiky durian fruit"], "榴莲", "食品", 5),
    (["cherries", "a few cherries"], "樱桃", "食品", 5),
    (["plums", "a few plums"], "李子", "食品", 7),
    (["jujube fruits", "a few red dates"], "枣", "食品", 30),
    (["a pineapple", "a whole pineapple"], "菠萝", "食品", 10),
    (["a coconut", "a brown coconut"], "椰子", "食品", 30),
    (["lychees", "a few lychee fruits"], "荔枝", "食品", 5),
    # 蔬菜细分：菜篮/购物车里大量散装蔬菜无包装文字，YOLO(无此类)与
    # OCR(无文字)都抓不到，只能靠 CLIP 区分品种；统称"蔬菜"太粗，用户要的是
    # 具体品类（番茄/青椒…）。下面这些与"蔬菜"统称并存，由 region_cls 的
    # 统称细化逻辑在得分接近时优先取具体品类。
    (["a red tomato", "fresh tomatoes on a vine"], "西红柿/番茄", "食品", 7),
    (["a green bell pepper", "fresh green chili peppers"], "青椒/辣椒", "食品", 7),
    (["a cucumber", "fresh cucumbers"], "黄瓜", "食品", 7),
    (["potatoes", "a few potatoes"], "土豆", "食品", 30),
    (["onions", "a few onions"], "洋葱", "食品", 30),
    (["carrots", "fresh carrots"], "胡萝卜", "食品", 14),
    (["broccoli", "a head of broccoli"], "西兰花", "食品", 5),
    (["chinese cabbage", "a head of napa cabbage"], "大白菜", "食品", 7),
    (["eggplants", "a purple eggplant"], "茄子", "食品", 5),
    (["corn cobs", "fresh sweet corn"], "玉米", "食品", 5),
    (["a bottled drink", "a plastic bottle of beverage"], "瓶装饮品", "食品", 365),
    (["a package of dried spaghetti pasta", "spaghetti pasta in a plastic bag",
      "uncooked pasta"], "意大利面", "食品", 730),
    (["dried Chinese noodle strands", "a pack of dried noodles"], "挂面/面条", "食品", 365),
    (["a bottle of shampoo", "shampoo in a plastic bottle"], "洗发水", "日化", 730),
    (["a bottle of body wash", "body wash in a bottle"], "沐浴露", "日化", 730),
    (["a tube of toothpaste", "toothpaste with a box"], "牙膏", "日化", 730),
    (["a bottle of dishwashing liquid", "dishwashing liquid in a bottle"], "洗洁精", "日化", 730),
    (["a bottle of laundry detergent", "laundry detergent in a plastic bottle"], "洗衣液", "日化", 730),
    (["a pack of facial tissues", "a box of tissues"], "纸巾", "日化", 1095),
    (["a box of disposable face masks", "face masks in a box"], "口罩", "日化", 730),
    (["a bottle of skin care lotion", "a jar of face cream"], "护肤品", "日化", 1095),
    (["a toothbrush", "a toothbrush in a cup"], "牙刷", "日化", 180),
    (["a box of cold and flu medicine", "medicine tablets in a blister pack"], "感冒药", "药品", 730),
    (["a bottle of vitamin supplements", "vitamin pills in a bottle"], "维生素", "药品", 730),
    (["a tube of medicinal ointment", "a first aid ointment tube"], "外用药膏", "药品", 730),
    (["a pack of adhesive bandages", "band-aids in a box"], "创可贴", "药品", 1095),
    (["a bottle of oral liquid medicine", "a small bottle of cough syrup"], "口服液", "药品", 540),
    (["a notebook", "a spiral notebook"], "笔记本", "文具", 3650),
    (["a pen", "a ballpoint pen"], "签字笔", "文具", 3650),
    (["pencils", "a few wooden pencils"], "铅笔", "文具", 3650),
    (["an eraser", "a rubber eraser"], "橡皮", "文具", 3650),
    (["a ruler", "a plastic ruler"], "尺子", "文具", 3650),
    (["a pair of scissors", "office scissors"], "剪刀", "文具", 3650),
    (["a bottle of glue", "a glue stick"], "胶水", "文具", 730),
    (["a stack of printing paper", "a ream of A4 paper"], "打印纸", "文具", 3650),
    (["a paper folder", "a stack of file folders"], "文件夹", "文具", 3650),
]


def _all_prompts() -> List[str]:
    """展开成完整提示词列表：每条前加 'a photo of '。"""
    return [f"a photo of {d}" for descs, _, _, _ in PROMPT_ITEMS for d in descs]


def _prompt_to_item() -> List[int]:
    """提示词下标 -> 物品下标。"""
    out = []
    for i, (descs, _, _, _) in enumerate(PROMPT_ITEMS):
        out.extend([i] * len(descs))
    return out


def _prompts_hash() -> str:
    payload = json.dumps([[d for d, _, _, _ in PROMPT_ITEMS],
                          [n for _, n, _, _ in PROMPT_ITEMS]], ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- 加载

def _tune_threads() -> None:
    """把 torch 的 CPU 线程数调到本机可用核心（上限 8）。

    默认 torch 只开 4 线程（实测本机 8 核），ViT-L/14 单次前向 0.79s；
    调到 8 线程后 0.57s（-28%）。区域扫描/多物品切分要连续跑十几张
    裁剪图，这项收益会被放大。异常时静默忽略（只是性能优化）。
    """
    try:
        import os
        import torch
        n = max(1, min(8, (os.cpu_count() or 4)))
        if torch.get_num_threads() < n:
            torch.set_num_threads(n)
    except Exception:
        pass


def _ensure_loaded() -> bool:
    """懒加载模型与文本嵌入；任何一步失败都返回 False（优雅降级）。"""
    global _model, _processor, _text_emb
    if _model is not None and _text_emb is not None:
        return True
    with _lock:
        if _model is not None and _text_emb is not None:
            return True
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            _tune_threads()
            local = CLIP_DIR if (CLIP_DIR / "config.json").exists() else CLIP_MODEL_NAME
            _processor = CLIPProcessor.from_pretrained(local)
            _model = CLIPModel.from_pretrained(local)
            _model.eval()

            # 文本嵌入：有缓存且提示词未变则直接读，否则计算并落盘
            h = _prompts_hash()
            if TEXT_EMB_FILE.exists() and PROMPTS_FILE.exists():
                try:
                    if json.loads(PROMPTS_FILE.read_text(encoding="utf-8")).get("hash") == h:
                        _text_emb = torch.load(TEXT_EMB_FILE, map_location="cpu",
                                               weights_only=True)
                except Exception:
                    _text_emb = None
            if _text_emb is None:
                prompts = _all_prompts()
                inputs = _processor(text=prompts, return_tensors="pt", padding=True,
                                    truncation=True)
                with torch.no_grad():
                    feats = _model.text_projection(
                        _model.text_model(**inputs).pooler_output)
                _text_emb = torch.nn.functional.normalize(feats, dim=-1)
                try:
                    CLIP_DIR.mkdir(parents=True, exist_ok=True)
                    torch.save(_text_emb, TEXT_EMB_FILE)
                    PROMPTS_FILE.write_text(json.dumps({"hash": h}), encoding="utf-8")
                except Exception:
                    pass
            return True
        except Exception:
            _model, _processor, _text_emb = None, None, None
            return False


def available() -> bool:
    """模型是否已下载到本地（不触发下载，不加载权重）。"""
    return (CLIP_DIR / "config.json").exists()


def status() -> dict:
    return {
        "available": available(),
        "loaded": _model is not None,
        "download_state": _download_state["state"],
        "download_error": _download_state["error"],
        "model": CLIP_MODEL_NAME,
        "items": len(PROMPT_ITEMS),
    }


def download_model() -> bool:
    """下载模型权重到 CLIP_DIR（供设置页一键安装使用）。

    优先走 hf-mirror 镜像（部分网络环境直连 huggingface.co 会被代理拦截），
    镜像失败再回退官方端点。
    """
    global _model, _processor, _text_emb
    _download_state.update({"state": "downloading", "error": ""})
    last_err = ""
    for endpoint in ("https://hf-mirror.com", "https://huggingface.co"):
        try:
            import os
            os.environ["HF_ENDPOINT"] = endpoint
            os.environ["HF_HUB_DISABLE_XET"] = "1"   # 镜像的 Xet 通道会 401，走普通 HTTP
            from huggingface_hub import snapshot_download
            snapshot_download(CLIP_MODEL_NAME, local_dir=str(CLIP_DIR))
            with _lock:
                _model, _processor, _text_emb = None, None, None
            _download_state["state"] = "ready"
            return True
        except Exception as e:  # noqa: BLE001
            last_err = str(e)[:200]
    _download_state.update({"state": "failed", "error": last_err})
    return False


def download_model_async():
    """后台线程下载，避免阻塞请求。"""
    if _download_state["state"] == "downloading":
        return
    t = threading.Thread(target=download_model, daemon=True)
    t.start()


def warmup() -> bool:
    """加载模型并跑一次推理，验证可用。"""
    from PIL import Image
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    return _ensure_loaded() and classify(img) is not None


# ---------------------------------------------------------------- 推理

def classify(pil_image, top_k: int = 5) -> Optional[List[Dict]]:
    """
    零样本分类。返回 [{name, category, default_shelf_life_days, score}, ...]，
    按置信度降序；模型不可用时返回 None。

    每个物品有 2-3 个提示词（提示词集成），该物品得分取其最高分。
    """
    if not _ensure_loaded():
        return None
    try:
        import torch
        from PIL import Image

        if not isinstance(pil_image, Image.Image):
            return None
        img = pil_image.convert("RGB")
        inputs = _processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feats = _model.visual_projection(
                _model.vision_model(**inputs).pooler_output)
        img_emb = torch.nn.functional.normalize(feats, dim=-1)
        scores = (img_emb @ _text_emb.T) * 100.0     # logits
        probs = torch.softmax(scores, dim=-1)[0]

        # 提示词 -> 物品，按物品取最高分
        p2i = _prompt_to_item()
        best: Dict[int, float] = {}
        for idx, prob in enumerate(probs.tolist()):
            item = p2i[idx]
            if item not in best or prob > best[item]:
                best[item] = prob
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:top_k]
        out = []
        for item_idx, prob in ranked:
            _, name, cname, days = PROMPT_ITEMS[item_idx]
            out.append({"name": name, "category": cname,
                        "default_shelf_life_days": days,
                        "score": round(float(prob), 3)})
        return out
    except Exception:
        return None
