# -*- coding: utf-8 -*-
"""只重识别之前因 matched 未绑定异常失败的图片，合并进 JSON 并仅导入这些新草稿。"""
import datetime as dt
import io
import json
import sys
import uuid
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"E:\保质期\bzq\backend")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base
from app.services.pipeline import recognize_photo
from app import config
import urllib.request

SRC = Path(r"E:\保质期\物品素材")
OUT = Path(r"E:\保质期\bzq\backend\_batch_recognize.json")
API = "http://127.0.0.1:8000"

data = json.loads(OUT.read_text(encoding="utf-8"))
failed = [e for e in data if e.get("message", "").startswith("识别异常")]
print(f"失败条目: {len(failed)} 张", flush=True)

engine = create_engine("sqlite:///E:/保质期/bzq/backend/data/bzq.db")
Base.metadata.create_all(engine)
db = sessionmaker(bind=engine)()

for e in failed:
    sp = SRC / e["src"]
    if not sp.exists():
        print("源文件缺失，跳过:", e["src"], flush=True)
        continue
    sub = dt.datetime.now().strftime("%Y%m")
    dest_dir = config.UPLOAD_DIR / sub
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex[:16]}{sp.suffix.lower()}"
    dest = dest_dir / fname
    dest.write_bytes(sp.read_bytes())
    rel = f"{sub}/{fname}"
    try:
        res = recognize_photo(db, dest, rel, today=dt.date.today())
        e["image_url"] = f"/uploads/{rel}"
        e["message"] = res.get("message", "")
        e["image_width"] = res.get("image_width", 0)
        e["image_height"] = res.get("image_height", 0)
        e["drafts"] = [json.loads(d.model_dump_json()) for d in res["drafts"]]
        print(f"  {e['src']}: {len(e['drafts'])} 条 -> "
              f"{[d['name'] for d in e['drafts']]}", flush=True)
    except Exception as ex:
        e["message"] = f"识别异常: {ex}"
        print(f"  {e['src']} 仍失败: {ex}", flush=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

# 仅导入这些（现已修好）失败条目里的草稿
items, skipped = [], 0
for e in failed:
    for d in e.get("drafts", []):
        if not d.get("name") and not d.get("production_date") \
                and not d.get("expiry_date") and not d.get("bbox"):
            skipped += 1
            continue
        item = {
            "name": d.get("name") or "未识别物品",
            "raw_name": d.get("raw_name") or "",
            "category_id": d.get("category_id"),
            "quantity": d.get("quantity") or 1,
            "unit": d.get("unit") or "件",
            "image_path": e["image_url"].lstrip("/").split("uploads/")[-1],
            "bbox": json.dumps([int(v) for v in d["bbox"]]) if d.get("bbox") else "",
            "production_date": d.get("production_date"),
            "shelf_life_days": d.get("shelf_life_days"),
            "expiry_date": d.get("expiry_date"),
            "date_text": d.get("date_text") or "",
            "date_source": d.get("date_source") or "manual",
            "confidence": d.get("confidence"),
            "needs_review": bool(d.get("needs_review")),
            "note": (d.get("note") or "") or "",
        }
        dn = d.get("date_note") or ""
        if dn and dn not in item["note"]:
            item["note"] = (item["note"] + " " + dn).strip()
        item["note"] = (item["note"] + f" [来源:{e['src']}]").strip()
        items.append(item)

print(f"\n待入库（仅失败补识别）{len(items)} 条（跳过空草稿 {skipped} 条）", flush=True)
created, failed_batches = [], []
B = 20
for s in range(0, len(items), B):
    chunk = items[s:s + B]
    req = urllib.request.Request(
        API + "/api/items",
        data=json.dumps({"items": chunk}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read().decode("utf-8"))
            created.extend(o["id"] for o in out)
            print(f"  批次 {s//B+1}: 入库 {len(out)} 条 OK", flush=True)
    except Exception as ex:
        body = ex.read().decode("utf-8", "ignore")[:300] if hasattr(ex, "read") else ""
        failed_batches.append((s // B + 1, str(ex), body))
        print(f"  批次 {s//B+1} 失败: {ex} {body}", flush=True)

print(f"\n补录完成: 成功 {len(created)} 条，失败 {len(failed_batches)} 批", flush=True)
if created:
    print(f"新物品 ID: {created[0]} .. {created[-1]}", flush=True)
json.dump({"created": created, "failed": failed_batches},
          open(r"E:\保质期\bzq\backend\_fix_import_done.json", "w", encoding="utf-8"),
          ensure_ascii=False)
print("done", flush=True)
