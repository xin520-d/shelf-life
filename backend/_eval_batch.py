# -*- coding: utf-8 -*-
"""对照 标准答案.jsonl 评估批量识别准确率。"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(r"E:\保质期\物品素材")
rec = json.loads(Path(r"E:\保质期\bzq\backend\_batch_recognize.json")
                 .read_text(encoding="utf-8"))
truth = {}
for line in (BASE / "标准答案.jsonl").read_text(
        encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        truth[d["image"]] = d["objects"]


def norm_name(s: str) -> str:
    return (s or "").replace(" ", "").strip()


def name_hit(rec_name: str, gt_name: str) -> bool:
    a, b = norm_name(rec_name), norm_name(gt_name)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def date_hit(rec_date, gt_date) -> bool:
    return bool(rec_date) and str(rec_date)[:10] == str(gt_date)[:10]


# 汇总
by_src = {}
for e in rec:
    key = Path(e["src"]).name
    by_src[key] = e

rows = []
tot_gt = tot_name = tot_prod = tot_exp = 0
miss_imgs = []
for img, objs in sorted(truth.items()):
    e = by_src.get(img)
    drafts = e["drafts"] if e else []
    # 简单匹配：按顺序 + 名称包含关系
    used = set()
    for o in objs:
        tot_gt += 1
        best = None
        for i, d in enumerate(drafts):
            if i in used:
                continue
            if name_hit(d.get("name", ""), o["product_name"]) \
                    or name_hit(d.get("raw_name", ""), o["product_name"]):
                best = (i, d)
                break
        if best is None and len(drafts) == len(objs):
            # 名称对不上时按位置兜底：第 k 个物品配第 k 条草稿
            idx = objs.index(o)
            if idx < len(drafts) and idx not in used:
                best = (idx, drafts[idx])
        if best:
            i, d = best
            used.add(i)
            tot_name += 1
            if date_hit(d.get("production_date"), o.get("production_date")):
                tot_prod += 1
            if date_hit(d.get("expiry_date"), o.get("expiry_date")):
                tot_exp += 1
            rows.append((img, o["product_name"], d.get("name", ""),
                         o.get("production_date"),
                         str(d.get("production_date") or "-")[:10],
                         o.get("expiry_date"),
                         str(d.get("expiry_date") or "-")[:10]))
        else:
            rows.append((img, o["product_name"], "<未匹配>",
                         o.get("production_date"), "-",
                         o.get("expiry_date"), "-"))
    if not drafts:
        miss_imgs.append(img)

print(f"标准答案物品总数: {tot_gt}")
print(f"识别出名称(含部分匹配): {tot_name} ({tot_name/tot_gt*100:.0f}%)")
print(f"生产日期正确: {tot_prod} ({tot_prod/tot_gt*100:.0f}%)")
print(f"到期日正确:   {tot_exp} ({tot_exp/tot_gt*100:.0f}%)")
print(f"\n零草稿图片: {len(miss_imgs)} 张 {miss_imgs}")
print("\n--- 明细（GT名 / 识别名 / GT生产日 / 识别生产日 / GT到期 / 识别到期）---")
for r in rows:
    print(f"{r[0][:20]:22} | {r[1][:14]:16} | {str(r[2])[:14]:16} | "
          f"{str(r[3])[:10]:10} | {str(r[4])[:10]:10} | "
          f"{str(r[5])[:10]:10} | {str(r[6])[:10]}")
