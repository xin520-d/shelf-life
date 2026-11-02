"""后端端到端冒烟测试：识别 -> 入库 -> 修正记忆 -> 统计 -> 导出。"""
import json
import urllib.request
import urllib.parse

BASE = "http://127.0.0.1:8000"
BOUNDARY = "----testboundary"


def post_file(url, filepath):
    with open(filepath, "rb") as f:
        content = f.read()
    body = (f"--{BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="test.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n").encode() + content + \
           f"\r\n--{BOUNDARY}--\r\n".encode()
    req = urllib.request.Request(BASE + url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"})
    return json.load(urllib.request.urlopen(req))


def api(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def download(path, out):
    urllib.request.urlretrieve(BASE + path, out)
    import os
    return os.path.getsize(out)


print("== 1. 识别照片 ==")
results = post_file("/api/recognize", "E:/保质期/bzq/backend/data/test_fridge.jpg")
drafts = results[0]["drafts"]
print(f"   草稿数: {len(drafts)}")

print("== 2. 修正并入库（药品草稿统一改名=感冒灵颗粒/药品） ==")
items = []
for d in drafts:
    # 药品草稿现已被关键词自动命名为"感冒药"，此处统一改写以验证用户修正链路
    if not d["name"] or d["category_name"] == "药品":
        d["name"] = "感冒灵颗粒"
        d["category_id"] = 3  # 药品
    # 模拟一条 YOLO 检测结果（raw_name=bottle），用于验证修正记忆
    raw_name = "bottle" if d["name"] == "矿泉水" else d["raw_name"]
    items.append({
        "name": d["name"], "raw_name": raw_name, "category_id": d["category_id"],
        "location_id": 1, "quantity": d["quantity"], "unit": d["unit"],
        "image_path": results[0]["image_url"].replace("/uploads/", ""),
        "bbox": json.dumps(d["bbox"]),
        "production_date": d["production_date"], "shelf_life_days": d["shelf_life_days"],
        "expiry_date": d["expiry_date"], "date_text": d["date_text"],
        "date_source": d["date_source"], "confidence": d["confidence"],
        "needs_review": d["needs_review"],
    })
created = api("POST", "/api/items", {"items": items})
print(f"   入库: {len(created)} 条; 缩略图: {[c['thumb_url'] for c in created]}")

print("== 3. 物品列表（按紧急度排序） ==")
lst = api("GET", "/api/items?status=active")
for it in lst:
    print(f"   [{it['expiry_status']:8s}] {it['name']:10s} 剩余{it['remaining_days']}天 到期{it['expiry_date']}")

print("== 4. 编辑 + 修正记忆 ==")
target = next(i for i in lst if i["name"] == "感冒灵颗粒")
patched = api("PATCH", f"/api/items/{target['id']}",
              {"location_id": 6, "note": "家庭药箱"})
print(f"   PATCH ok: {patched['name']} -> 位置 {patched['location_name']}")
# 把"矿泉水"(raw_name=bottle) 从食品改为日化 -> 应写入修正记忆
water = next(i for i in lst if i["name"] == "矿泉水")
api("PATCH", f"/api/items/{water['id']}", {"category_id": 2})
corr = api("GET", "/api/corrections")
print(f"   修正记忆: {[(c['raw_name'], c['category_name']) for c in corr]}")
assert any(c["raw_name"] == "bottle" and c["category_name"] == "日化" for c in corr), "修正记忆未生效"
print("   修正记忆验证通过")

print("== 5. 消耗一条 ==")
bread = next(i for i in lst if "面包" in i["name"])
print("   ", api("POST", f"/api/items/{bread['id']}/finish?action=consume"))

print("== 6. 统计看板 ==")
s = api("GET", "/api/stats")
print(f"   在册 {s['total_active']} | 临期 {s['near_expiry']} | 过期 {s['expired']} | 正常 {s['normal']} | 未知 {s['unknown']}")
print(f"   7天内 {s['due_7d']} | 30天内 {s['due_30d']}")
print(f"   类别分布: {[(c['name'], c['count']) for c in s['by_category']]}")
print(f"   位置分布: {[(l['name'], l['count']) for l in s['by_location']]}")

print("== 7. 导出 ==")
csv_size = download("/api/export/csv", "E:/保质期/bzq/backend/data/test_export.csv")
xlsx_size = download("/api/export/xlsx", "E:/保质期/bzq/backend/data/test_export.xlsx")
print(f"   CSV {csv_size} bytes, XLSX {xlsx_size} bytes")

print("== 8. 缩略图检查 ==")
import os
thumbs = os.listdir("E:/保质期/bzq/backend/uploads/thumbs") if os.path.exists("E:/保质期/bzq/backend/uploads/thumbs") else []
print(f"   thumbs: {thumbs}")

print("\nALL BACKEND E2E TESTS PASSED")
