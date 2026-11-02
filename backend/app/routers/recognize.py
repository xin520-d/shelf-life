"""识别路由：照片上传 + 识别流水线（单张 / 批量，模块 1/3/9）。"""
import datetime as dt
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from ..database import get_db
from .. import config
from ..services.pipeline import recognize_photo

router = APIRouter(prefix="/api/recognize", tags=["recognize"])

ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@router.post("", response_model=List[dict])
async def recognize(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """
    一次上传一张或多张照片，逐张执行 检测 -> OCR -> 日期归一化 -> 归类。
    返回识别结果（草稿），待前端确认后调用 POST /api/items 入库。
    """
    results: List[dict] = []
    today = dt.date.today()
    for f in files:
        suffix = Path(f.filename or "photo.jpg").suffix.lower()
        if suffix not in ALLOWED:
            suffix = ".jpg"
        # uploads/YYYYMM/uuid.jpg
        sub = dt.datetime.now().strftime("%Y%m")
        dest_dir = config.UPLOAD_DIR / sub
        dest_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{uuid.uuid4().hex[:16]}{suffix}"
        dest = dest_dir / fname
        content = await f.read()
        dest.write_bytes(content)
        rel = f"{sub}/{fname}"

        try:
            result = recognize_photo(db, dest, rel, today=today)
        except Exception as e:  # 单张失败不影响整批
            # 注意：字段必须与 PhotoResult 完全一致（含 clip_top/is_screenshot），
            # 否则前端访问缺失字段会整个白屏
            result = {
                "filename": f.filename or fname,
                "image_url": f"/uploads/{rel}",
                "image_width": 0, "image_height": 0,
                "ocr_texts": [], "drafts": [],
                "clip_top": [], "is_screenshot": False,
                "message": f"识别失败：{e}",
            }
        results.append(result)
    return results
