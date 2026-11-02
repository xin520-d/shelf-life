# -*- coding: utf-8 -*-
"""按「日期写法类型」统计当前系统的日期解析准确率。

数据源（都是系统里现成的真实数据，不造数字）：
  - 测试素材：E:\\保质期\\物品素材\\{1件商品,2件商品,3件商品}\\s0*_*item*.png（60 张）
  - 标准答案：E:\\保质期\\物品素材\\标准答案.jsonl（92 个物品）
  - 识别：直接调用线上同一个 app.services.pipeline.recognize_photo（全流程 YOLO+OCR+CLIP+点阵）

写法类型归并（标准答案是 6 种细分类，汇报按 3 类口径）：
  直接日期型   ymd_slash / ymd_cn / compact8 / en_pair（包装上直接印了日期）
  相对保质期型 label_spray（只印生产日期 + 保质期时长，到期日需推算）
  截止日期型   expire_only（直接印「保质期至 20xx.xx」）

用法：python _eval_date_accuracy.py
输出：_eval_date_accuracy.json（明细）+ 控制台汇总
"""
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(r"E:\保质期\物品素材")
OUT = Path(__file__).with_name("_eval_date_accuracy.json")

STYLE_GROUP = {
    "ymd_slash": "直接日期型",
    "ymd_cn": "直接日期型",
    "compact8": "直接日期型",
    "en_pair": "直接日期型",
    "label_spray": "相对保质期型",
    "expire_only": "截止日期型",
}


def load_truth():
    rows = [json.loads(l) for l in (BASE / "标准答案.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    items = []
    for r in rows:
        img = BASE / r.get("folder", "") / r["image"] if r.get("folder") else None
        if not img or not img.exists():
            # 文件名本身已含类别：按目录扫描兜底
            cand = list(BASE.rglob(r["image"]))
            img = cand[0] if cand else None
        for o in r["objects"]:
            items.append({"image": r["image"], "path": img, "gt": o})
    return items


def _g(d, key, default=None):
    """草稿既可能是 dict 也可能是 pydantic 模型，统一取值。"""
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def center_inside(inner, outer):
    x1, y1, x2, y2 = outer
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return x1 <= cx <= x2 and y1 <= cy <= y2


def match_draft(gt_box, drafts, used):
    """GT 框与草稿框的最佳匹配：优先 IoU，其次「草稿框中心落在 GT 框内」。"""
    best, best_score = None, 0.0
    for i, d in enumerate(drafts):
        if i in used or not _g(d, "bbox"):
            continue
        score = iou(gt_box, _g(d, "bbox"))
        if score > best_score:
            best, best_score = i, score
    if best is None or best_score < 0.15:
        for i, d in enumerate(drafts):
            if i in used or not _g(d, "bbox"):
                continue
            if center_inside(_g(d, "bbox"), gt_box):
                best, best_score = i, 0.14
                break
    if best is None:
        return None
    used.add(best)
    return drafts[best]


def main():
    from app.database import SessionLocal
    from app.services.pipeline import recognize_photo

    truth = load_truth()
    by_image = defaultdict(list)
    for it in truth:
        by_image[it["image"]].append(it)

    details = []
    t0 = time.time()
    for k, (img_name, objs) in enumerate(sorted(by_image.items()), 1):
        path = objs[0]["path"]
        if not path or not path.exists():
            for it in objs:
                details.append({"image": img_name, "style": it["gt"].get("date_style"),
                                "name": it["gt"]["product_name"], "matched": False,
                                "error": "素材缺失"})
            continue
        db = SessionLocal()
        try:
            res = recognize_photo(db, path, f"eval/{img_name}", today=__import__("datetime").date(2026, 10, 21))
            drafts = res.get("drafts", [])
        except Exception as e:  # 单图失败不影响整体
            drafts = []
            print(f"  !! {img_name} 识别异常: {type(e).__name__}: {e}")
        finally:
            db.close()

        used = set()
        for it in objs:
            gt = it["gt"]
            d = match_draft([int(v) for v in gt["box"]], drafts, used) if gt.get("box") else None
            exp = str(gt.get("expiry_date") or "")[:10]
            rec_exp = str(_g(d, "expiry_date") or "")[:10]
            rec_prod = str(_g(d, "production_date") or "")[:10]
            details.append({
                "image": img_name,
                "style": gt.get("date_style"),
                "group": STYLE_GROUP.get(gt.get("date_style"), "其他"),
                "name": gt["product_name"],
                "gt_prod": gt.get("production_date"),
                "gt_shelf": gt.get("shelf_life_text"),
                "gt_exp": exp,
                "rec_name": _g(d, "name"),
                "rec_prod": rec_prod,
                "rec_exp": rec_exp,
                "rec_source": _g(d, "date_source"),
                "needs_review": _g(d, "needs_review"),
                "matched": d is not None,
                "exp_ok": bool(exp) and rec_exp == exp,
                "prod_ok": bool(gt.get("production_date")) and rec_prod == str(gt["production_date"])[:10],
            })
        print(f"[{k}/{len(by_image)}] {img_name} drafts={len(drafts)} "
              f"({time.time() - t0:.0f}s)")

    # 汇总
    groups = defaultdict(lambda: {"n": 0, "ok": 0, "prod_ok": 0, "review": 0, "miss": 0})
    for r in details:
        g = groups[r["group"]]
        g["n"] += 1
        g["ok"] += 1 if r["exp_ok"] else 0
        g["prod_ok"] += 1 if r["prod_ok"] else 0
        g["review"] += 1 if r.get("needs_review") else 0
        g["miss"] += 0 if r["matched"] else 1

    total = {k: {"n": 0, "ok": 0, "prod_ok": 0, "review": 0, "miss": 0} for k in
             ("直接日期型", "相对保质期型", "截止日期型", "其他")}
    for name, g in groups.items():
        for kk in total[name]:
            total[name][kk] += g[kk]

    print("\n=== 按写法类型 ===")
    for name in ("直接日期型", "相对保质期型", "截止日期型"):
        g = total[name]
        rate = g["ok"] / g["n"] * 100 if g["n"] else 0
        print(f"{name}: 样本 {g['n']} 正确 {g['ok']} 准确率 {rate:.1f}% "
              f"(生产日期对 {g['prod_ok']}, 待复核 {g['review']}, 未匹配 {g['miss']})")
    n = sum(total[name]["n"] for name in total)
    ok = sum(total[name]["ok"] for name in total)
    print(f"合计: {n} 样本 / {ok} 正确 = {ok / n * 100:.1f}%")

    # 失败原因归类
    fails = [r for r in details if not r["exp_ok"]]
    print(f"\n解析失败 {len(fails)} 条（待人工复核 {sum(1 for r in fails if r.get('needs_review'))} 条）")
    for r in fails[:25]:
        print(f"  {r['image'][:16]:18} {r['group']:6} {r['name'][:12]:14} "
              f"GT {r['gt_exp'] or '-':10} → 识别 {r['rec_exp'] or '-':10} "
              f"review={r.get('needs_review')}")

    OUT.write_text(json.dumps({"groups": total, "details": details},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细已写入 {OUT}")


if __name__ == "__main__":
    main()
