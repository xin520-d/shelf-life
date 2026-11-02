# -*- coding: utf-8 -*-
"""幂等补导入：仅导入数据库中尚未存在的来源（按 [来源:src] 备注去重）。"""
import io
import json
import sys
import urllib.request
from app.database import SessionLocal
from app.models import Item

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

API = "http://127.0.0.1:8000"
OUT = r"E:\保质期\bzq\backend\_batch_recognize.json"

# 已入库来源
db = SessionLocal()
have = set()
for it in db.query(Item).all():
    n = it.note or ""
    if "[来源:" in n:
        have.add(n.split("[来源:")[1].rstrip("]"))
db.close()

data = json.loads(open(OUT, encoding="utf-8").read())
items, skipped = [], 0
for e in data:
    if e["src"] in have:
        continue
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

print(f"待补导入 {len(items)} 条（跳过空草稿 {skipped} 条）", flush=True)
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

print(f"\n补导入完成: 成功 {len(created)} 条，失败 {len(failed_batches)} 批", flush=True)
if created:
    print(f"新物品 ID: {created[0]} .. {created[-1]}", flush=True)
json.dump({"created": created, "failed": failed_batches},
          open(r"E:\保质期\bzq\backend\_import_missing_done.json", "w", encoding="utf-8"),
          ensure_ascii=False)
print("done", flush=True)
