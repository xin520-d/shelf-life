"""元数据路由：类别、存放位置、系统设置、修正记忆（模块 2 的映射表管理）。"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Category, Item, Setting, CorrectionMemory
from ..schemas import (CategoryIn, CategoryOut, SettingIn, CorrectionOut)

router = APIRouter(prefix="/api", tags=["meta"])


# ---------- 类别 ----------
@router.get("/categories", response_model=List[CategoryOut])
def list_categories(db: Session = Depends(get_db)):
    out = []
    for c in db.query(Category).order_by(Category.id).all():
        cnt = db.query(Item).filter(Item.category_id == c.id,
                                    Item.status == "active").count()
        out.append(CategoryOut(id=c.id, name=c.name, color=c.color,
                               default_shelf_life_days=c.default_shelf_life_days,
                               is_builtin=c.is_builtin, item_count=cnt))
    return out


@router.post("/categories", response_model=CategoryOut)
def create_category(payload: CategoryIn, db: Session = Depends(get_db)):
    if db.query(Category).filter(Category.name == payload.name).first():
        raise HTTPException(400, "类别已存在")
    c = Category(**payload.model_dump(), is_builtin=False)
    db.add(c)
    db.commit()
    db.refresh(c)
    return CategoryOut(id=c.id, name=c.name, color=c.color,
                       default_shelf_life_days=c.default_shelf_life_days,
                       is_builtin=False, item_count=0)


@router.patch("/categories/{cid}", response_model=CategoryOut)
def update_category(cid: int, payload: CategoryIn, db: Session = Depends(get_db)):
    c = db.get(Category, cid)
    if not c:
        raise HTTPException(404, "类别不存在")
    for k, v in payload.model_dump().items():
        setattr(c, k, v)
    db.commit()
    db.refresh(c)
    cnt = db.query(Item).filter(Item.category_id == c.id, Item.status == "active").count()
    return CategoryOut(id=c.id, name=c.name, color=c.color,
                       default_shelf_life_days=c.default_shelf_life_days,
                       is_builtin=c.is_builtin, item_count=cnt)


@router.delete("/categories/{cid}")
def delete_category(cid: int, db: Session = Depends(get_db)):
    c = db.get(Category, cid)
    if not c:
        raise HTTPException(404, "类别不存在")
    if c.is_builtin:
        raise HTTPException(400, "内置类别不可删除")
    other = db.query(Category).filter(Category.name == "其他").first()
    db.query(Item).filter(Item.category_id == cid).update(
        {"category_id": other.id if other else None})
    db.delete(c)
    db.commit()
    return {"ok": True}


# ---------- 设置 ----------
@router.get("/settings")
def get_settings(db: Session = Depends(get_db)):
    s = db.get(Setting, 1)
    return {
        "near_expiry_days": s.near_expiry_days if s else 7,
        "low_conf": s.low_conf if s and s.low_conf is not None else 0.6,
    }


@router.put("/settings")
def update_settings(payload: SettingIn, db: Session = Depends(get_db)):
    s = db.get(Setting, 1)
    if not s:
        s = Setting(id=1)
        db.add(s)
    s.near_expiry_days = payload.near_expiry_days
    s.low_conf = payload.low_conf
    db.commit()
    return {
        "near_expiry_days": s.near_expiry_days,
        "low_conf": s.low_conf,
    }


# ---------- 修正记忆 ----------
@router.get("/corrections", response_model=List[CorrectionOut])
def list_corrections(db: Session = Depends(get_db)):
    out = []
    for m in db.query(CorrectionMemory).order_by(CorrectionMemory.last_used_at.desc()).all():
        cat = db.get(Category, m.category_id)
        out.append(CorrectionOut(id=m.id, raw_name=m.raw_name, item_name=m.item_name,
                                 category_name=cat.name if cat else "",
                                 use_count=m.use_count, last_used_at=m.last_used_at))
    return out


@router.delete("/corrections/{mid}")
def delete_correction(mid: int, db: Session = Depends(get_db)):
    m = db.get(CorrectionMemory, mid)
    if not m:
        raise HTTPException(404, "记录不存在")
    db.delete(m)
    db.commit()
    return {"ok": True}


# ---------- 识别模型状态（YOLO / RapidOCR / CLIP） ----------
@router.get("/models")
def list_models():
    """三条识别通道的状态，供设置页展示。"""
    from ..services import clip_cls, ocr
    from ..services.detector import model_ready
    return {
        "yolo": {"name": "YOLO11n", "ready": model_ready(),
                 "desc": "通用物体检测（COCO 80 类）"},
        "ocr": {"name": "RapidOCR", "ready": ocr.ocr_ready(),
                "desc": "中文包装文字识别（多尺度）"},
        "clip": {"name": clip_cls.CLIP_MODEL_NAME, "ready": clip_cls.available(),
                 "desc": f"零样本视觉分类（{len(clip_cls.PROMPT_ITEMS)} 类常见物品）",
                 **{k: v for k, v in clip_cls.status().items()
                    if k in ("download_state", "download_error")}},
    }


@router.post("/models/clip/download")
def download_clip_model():
    """一键安装 CLIP 视觉识别模型（后台线程下载，约 1.7GB）。"""
    from ..services import clip_cls
    if clip_cls.available():
        return {"ok": True, "message": "模型已安装"}
    clip_cls.download_model_async()
    return {"ok": True, "message": "已开始下载，完成后自动生效"}
