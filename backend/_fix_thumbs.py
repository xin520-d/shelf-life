# -*- coding: utf-8 -*-
"""修复 bbox 存储格式（逗号分隔 -> JSON 数组）并补生成缺失缩略图。"""
import io
import json
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"E:\保质期\bzq\backend")

from app.database import SessionLocal
from app.models import Item
from app.routers.items import _save_thumbnail, _thumb_path
from app import config

db = SessionLocal()
items = db.query(Item).order_by(Item.id).all()
print(f"库内物品总数: {len(items)}", flush=True)

# 1) 修 bbox 格式
fixed_bbox, bad_bbox = 0, []
for it in items:
    b = (it.bbox or "").strip()
    if not b:
        continue
    ok = False
    try:
        v = json.loads(b)
        ok = isinstance(v, list) and len(v) == 4
    except Exception:
        ok = False
    if ok:
        continue
    parts = [p.strip() for p in b.replace("，", ",").split(",") if p.strip()]
    try:
        nums = [int(float(p)) for p in parts]
        if len(nums) == 4:
            it.bbox = json.dumps(nums)
            fixed_bbox += 1
        else:
            bad_bbox.append((it.id, b))
    except Exception:
        bad_bbox.append((it.id, b))
db.commit()
print(f"bbox 格式修复: {fixed_bbox} 条，无法解析: {len(bad_bbox)} 条", flush=True)
if bad_bbox:
    print("  无法解析示例:", bad_bbox[:10], flush=True)

# 2) 补缩略图
made, failed = 0, []
for it in items:
    db.refresh(it)
    tp = Path(_thumb_path(it.id))
    if tp.exists():
        continue
    ok = _save_thumbnail(it)
    if ok:
        made += 1
    else:
        src = config.UPLOAD_DIR / it.image_path if it.image_path else None
        reason = "无image_path" if not it.image_path else (
            "无bbox" if not it.bbox else (
                "原图缺失" if not (src and os.path.exists(src)) else "裁剪失败"))
        failed.append((it.id, reason, it.image_path))

print(f"\n新生成缩略图: {made} 张，仍失败: {len(failed)} 条", flush=True)
if failed:
    from collections import Counter
    print("  失败原因分布:", dict(Counter(r for _, r, _ in failed)), flush=True)
    for f in failed[:15]:
        print("   ", f, flush=True)

# 3) 核对最终缩略图总数
thumb_dir = Path(config.UPLOAD_DIR) / "thumbs"
cnt = len(list(thumb_dir.glob("item_*.jpg"))) if thumb_dir.exists() else 0
print(f"\n缩略图目录现有: {cnt} 张 / 物品 {len(items)} 件", flush=True)
db.close()
print("done", flush=True)
