# -*- coding: utf-8 -*-
"""把 _batch_recognize.json 的草稿经 POST /api/items 批量入库。"""
import io
import json
import sys
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

API = "http://127.0.0.1:8000"
data = json.loads(open(r"E:\保质期\bzq\backend\_batch_recognize.json",
                       encoding="utf-8").read())

items = []
skipped = 0
for e in data:
    for d in e["drafts"]:
        # 过滤完全空名的草稿（无名称无日期无框的纯噪声）
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
            "image_path": e["image_url"].lstrip("/").split("uploads/")[-1]
                          if "uploads/" in e["image_url"] else "",
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
        # 把机器建议说明并进 note
        dn = d.get("date_note") or ""
        if dn and dn not in item["note"]:
            item["note"] = (item["note"] + " " + dn).strip()
        # 附带来源文件名，方便回溯
        item["note"] = (item["note"] + f" [来源:{e['src']}]").strip()
        items.append(item)

print(f"待入库 {len(items)} 条（跳过空草稿 {skipped} 条）", flush=True)

created, failed = [], []
B = 20  # 每批 20 条
for s in range(0, len(items), B):
    chunk = items[s:s + B]
    req = urllib.request.Request(
        API + "/api/items",
        data=json.dumps({"items": chunk}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read().decode("utf-8"))
            created.extend(o["id"] for o in out)
            print(f"  批次 {s//B+1}: 入库 {len(out)} 条 OK", flush=True)
    except Exception as ex:
        body = ""
        if hasattr(ex, "read"):
            body = ex.read().decode("utf-8", "ignore")[:200]
        failed.append((s // B + 1, str(ex), body))
        print(f"  批次 {s//B+1} 失败: {ex} {body}", flush=True)

print(f"\n入库完成: 成功 {len(created)} 条，失败 {len(failed)} 批", flush=True)
if created:
    print(f"新物品 ID: {created[0]} .. {created[-1]}", flush=True)
Path_out = r"E:\保质期\bzq\backend\_batch_import_done.json"
json.dump({"created": created, "failed": failed},
          open(Path_out, "w", encoding="utf-8"), ensure_ascii=False)
print("done", flush=True)
