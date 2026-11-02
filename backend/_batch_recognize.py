# -*- coding: utf-8 -*-
"""批量识别 E:\\保质期\\物品素材 下所有图片（含子目录），草稿写入 JSON。
流程与 POST /api/recognize 完全一致：存 uploads/YYYYMM/uuid.ext -> recognize_photo。
"""
import datetime as dt
import io
import json
import sys
import time
import uuid
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"E:\保质期\bzq\backend")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base
from app.services.pipeline import recognize_photo
from app import config

SRC = Path(r"E:\保质期\物品素材")
OUT = Path(r"E:\保质期\bzq\backend\_batch_recognize.json")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

imgs = sorted(p for p in SRC.rglob("*")
              if p.is_file() and p.suffix.lower() in IMG_EXT)
print(f"共 {len(imgs)} 张图片待识别", flush=True)

engine = create_engine("sqlite:///E:/保质期/bzq/backend/data/bzq.db")
Base.metadata.create_all(engine)
db = sessionmaker(bind=engine)()

results = []
t_all = time.time()
for i, p in enumerate(imgs, 1):
    t0 = time.time()
    sub = dt.datetime.now().strftime("%Y%m")
    dest_dir = config.UPLOAD_DIR / sub
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex[:16]}{p.suffix.lower()}"
    dest = dest_dir / fname
    dest.write_bytes(p.read_bytes())
    rel = f"{sub}/{fname}"
    entry = {"src": str(p.relative_to(SRC)).replace("\\", "/"),
             "filename": p.name, "image_url": f"/uploads/{rel}",
             "message": "", "drafts": []}
    try:
        res = recognize_photo(db, dest, rel, today=dt.date.today())
        entry["message"] = res.get("message", "")
        entry["image_width"] = res.get("image_width", 0)
        entry["image_height"] = res.get("image_height", 0)
        for d in res["drafts"]:
            entry["drafts"].append(json.loads(d.model_dump_json()))
    except Exception as e:
        entry["message"] = f"识别异常: {e}"
    results.append(entry)
    # 每张落盘一次，崩溃不丢进度
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    names = [d["name"] for d in entry["drafts"]]
    print(f"[{i}/{len(imgs)}] {p.name} ({time.time()-t0:.1f}s) "
          f"{len(names)} 条: {names}", flush=True)

print(f"全部完成，总耗时 {time.time()-t_all:.0f}s，"
      f"草稿总数 {sum(len(e['drafts']) for e in results)}", flush=True)
