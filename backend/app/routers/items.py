"""物品路由：批量入库、查询、编辑、消耗/丢弃（模块 1/2/4 的落库环节）。"""
import datetime as dt
import json
import os
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from PIL import Image
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Item, Category, Location, Setting
from ..schemas import ItemIn, ItemPatch, ItemOut, BulkItemsIn
from ..services.date_parser import expiry_status
from ..services.pipeline import remember_correction
from .. import config

router = APIRouter(prefix="/api/items", tags=["items"])


class _BatchIds(BaseModel):
    ids: List[int]


class _BatchFinish(BaseModel):
    ids: List[int]
    action: str = Field("discard", pattern="^(consume|discard)$")


def _get_near_days(db: Session) -> int:
    s = db.get(Setting, 1)
    return s.near_expiry_days if s else config.NEAR_EXPIRY_DAYS


def _thumb_path(item_id: int) -> str:
    return str(config.UPLOAD_DIR / "thumbs" / f"item_{item_id}.jpg")


def _safe_unlink(path) -> None:
    """删除文件；失败（文件不存在/被占用/环境禁止删除）时静默跳过，
    不中断调用方。"""
    try:
        os.remove(path)
    except (OSError, SystemExit):
        pass


def _save_thumbnail(item: Item) -> bool:
    """按 bbox 从原图裁剪物品缩略图（列表展示用），失败不影响主流程。"""
    if not item.image_path or not item.bbox:
        return False
    try:
        src = config.UPLOAD_DIR / item.image_path
        pil = Image.open(src).convert("RGB")
        bbox = json.loads(item.bbox)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = 16
        crop = pil.crop((max(0, x1 - pad), max(0, y1 - pad),
                         min(pil.width, x2 + pad), min(pil.height, y2 + pad)))
        if crop.width < 4 or crop.height < 4:
            return False
        crop.thumbnail((320, 320))
        thumb_dir = config.UPLOAD_DIR / "thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        crop.save(_thumb_path(item.id), quality=85)
        return True
    except Exception:
        return False


def _delete_item_files(item: Item, db: Session) -> None:
    """删除物品关联的磁盘文件。

    缩略图与本物品一一对应，直接删；原图按 image_path 引用计数，
    仅当没有其他物品引用同一张照片时才删（多物品识别时一张照片
    会生成多条物品记录，不能因为删其中一个就把共享原图删掉）。
    """
    tp = _thumb_path(item.id)
    if os.path.exists(tp):
        _safe_unlink(tp)
    if item.image_path:
        still_ref = (db.query(Item)
                     .filter(Item.id != item.id,
                             Item.image_path == item.image_path)
                     .first())
        if not still_ref:
            src = config.UPLOAD_DIR / item.image_path
            if os.path.exists(src):
                _safe_unlink(src)


def _to_out(item: Item, today: dt.date, near_days: int) -> ItemOut:
    status, remaining = expiry_status(item.expiry_date, today, near_days)
    thumb = f"/uploads/thumbs/item_{item.id}.jpg"
    thumb_url = thumb if os.path.exists(_thumb_path(item.id)) else None
    return ItemOut(
        id=item.id, name=item.name, raw_name=item.raw_name,
        category_id=item.category_id,
        category_name=item.category.name if item.category else "",
        category_color=item.category.color if item.category else "#6b7280",
        quantity=item.quantity, unit=item.unit, image_path=item.image_path,
        production_date=item.production_date, shelf_life_days=item.shelf_life_days,
        expiry_date=item.expiry_date, date_text=item.date_text,
        date_source=item.date_source, confidence=item.confidence,
        needs_review=item.needs_review, note=item.note, status=item.status,
        remaining_days=remaining, expiry_status=status,
        created_at=item.created_at, thumb_url=thumb_url,
    )


@router.post("", response_model=List[ItemOut])
def create_items(payload: BulkItemsIn, db: Session = Depends(get_db)):
    """前端确认后的草稿批量入库。"""
    today = dt.date.today()
    near = _get_near_days(db)
    created: list = []
    for d in payload.items:
        if not db.get(Category, d.category_id):
            raise HTTPException(400, f"类别不存在: {d.category_id}")
        item = Item(**d.model_dump())
        item.status = "active"
        item.created_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        db.add(item)
        db.flush()
        created.append(item)
    db.commit()
    for it in created:
        db.refresh(it)
        _save_thumbnail(it)
    return [_to_out(it, today, near) for it in created]


def _match_expiry_filter(o: ItemOut, expiring_within: Optional[int],
                         status_filter: Optional[str]) -> bool:
    """列表的有效期筛选（临期 / 已过期标签页共用）。

    关键点：「N 天内到期」只算**尚未过期**的物品。已过期物品剩余天数为负，
    天然满足 `remaining_days <= N`，不加下界就会把已过期物品全塞进临期列表
    （2026-10-21 用户反馈：临期页里全是「已过期 2972 天」的物品）。
    """
    if expiring_within is not None and o.status == "active":
        if (o.remaining_days is None or o.remaining_days < 0
                or o.remaining_days > expiring_within):
            return False
    if status_filter and o.expiry_status != status_filter:
        return False
    return True


@router.get("", response_model=List[ItemOut])
def list_items(
    status: str = Query("active"),
    category_id: Optional[int] = None,
    location_id: Optional[int] = None,
    q: Optional[str] = None,
    expiring_within: Optional[int] = Query(None, description="N 天内到期（不含已过期）"),
    expiry_status_filter: Optional[Literal["near", "expired", "normal", "unknown"]] = Query(
        None, alias="expiry_status",
        description="按有效期状态过滤：near / expired / normal / unknown"),
    sort: str = Query("expiry", description="expiry / created / name"),
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    size: int = Query(20, ge=1, le=200, description="每页条数"),
    response: Response = None,
    db: Session = Depends(get_db),
):
    today = dt.date.today()
    near = _get_near_days(db)
    query = db.query(Item)
    if status != "all":
        query = query.filter(Item.status == status)
    if category_id:
        query = query.filter(Item.category_id == category_id)
    if location_id:
        query = query.filter(Item.location_id == location_id)
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Item.name.like(like), Item.raw_name.like(like),
                                 Item.note.like(like)))
    items = query.all()

    out = []
    for it in items:
        o = _to_out(it, today, near)
        if not _match_expiry_filter(o, expiring_within, expiry_status_filter):
            continue
        out.append(o)

    if sort == "created":
        out.sort(key=lambda o: o.created_at or dt.datetime.min, reverse=True)
    elif sort == "name":
        out.sort(key=lambda o: o.name)
    else:  # expiry：未知日期排最后，其余按剩余天数升序（最紧急在前）
        out.sort(key=lambda o: (o.remaining_days is None,
                                o.remaining_days if o.remaining_days is not None else 0))

    total = len(out)
    start = (page - 1) * size
    page_out = out[start:start + size]
    if response is not None:
        response.headers["X-Total-Count"] = str(total)
        response.headers["X-Page"] = str(page)
        response.headers["X-Page-Size"] = str(size)
    return page_out


@router.post("/batch/confirm")
def batch_confirm(payload: _BatchIds, db: Session = Depends(get_db)):
    """批量确认：清除选中物品的待审核标记。"""
    items = db.query(Item).filter(Item.id.in_(payload.ids)).all()
    n = 0
    for it in items:
        it.needs_review = False
        n += 1
    db.commit()
    return {"updated": n}


@router.post("/batch/finish")
def batch_finish(payload: _BatchFinish, db: Session = Depends(get_db)):
    """批量登记消耗/丢弃（软删除，保留记录、可恢复）。"""
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    status = "consumed" if payload.action == "consume" else "discarded"
    items = db.query(Item).filter(Item.id.in_(payload.ids)).all()
    n = 0
    for it in items:
        it.status = status
        it.finished_at = now
        it.updated_at = now
        n += 1
    db.commit()
    return {"updated": n, "status": status}


@router.post("/batch/delete")
def batch_delete(payload: _BatchIds, db: Session = Depends(get_db)):
    """批量彻底删除（硬删除，不可恢复），并清理关联文件。"""
    items = db.query(Item).filter(Item.id.in_(payload.ids)).all()
    n = 0
    for it in items:
        _delete_item_files(it, db)
        db.delete(it)
        n += 1
    db.commit()
    return {"deleted": n}


@router.get("/{item_id}", response_model=ItemOut)
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(404, "物品不存在")
    return _to_out(item, dt.date.today(), _get_near_days(db))


@router.patch("/{item_id}", response_model=ItemOut)
def update_item(item_id: int, patch: ItemPatch, db: Session = Depends(get_db)):
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(404, "物品不存在")
    data = patch.model_dump(exclude_unset=True)

    if "category_id" in data and not db.get(Category, data["category_id"]):
        raise HTTPException(400, "类别不存在")
    if data.get("location_id") and not db.get(Location, data["location_id"]):
        raise HTTPException(400, "位置不存在")

    # 用户修正类别/名称 -> 写入修正记忆（模块 2：类别映射与用户修正记忆）
    if ("category_id" in data or "name" in data) and item.raw_name:
        new_cat = data.get("category_id", item.category_id)
        new_name = data.get("name", item.name)
        if new_cat != item.category_id or new_name != item.name:
            remember_correction(db, item.raw_name, new_cat, new_name)

    for k, v in data.items():
        setattr(item, k, v)
    item.updated_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    db.commit()
    db.refresh(item)
    return _to_out(item, dt.date.today(), _get_near_days(db))


@router.post("/{item_id}/finish")
def finish_item(item_id: int, action: str = Query("consume", pattern="^(consume|discard)$"),
                db: Session = Depends(get_db)):
    """消耗或丢弃登记，避免已处理物品继续出现在临期提醒中。"""
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(404, "物品不存在")
    item.status = "consumed" if action == "consume" else "discarded"
    item.finished_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    item.updated_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    db.commit()
    return {"ok": True, "status": item.status}


@router.post("/{item_id}/restore")
def restore_item(item_id: int, db: Session = Depends(get_db)):
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(404, "物品不存在")
    item.status = "active"
    item.finished_at = None
    db.commit()
    return {"ok": True, "status": "active"}


@router.delete("/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(404, "物品不存在")
    _delete_item_files(item, db)
    db.delete(item)
    db.commit()
    return {"ok": True}


@router.post("/cleanup")
def cleanup_orphan_files(db: Session = Depends(get_db)):
    """清理未被任何物品引用的孤儿图片。

    包括：上传后丢弃全部草稿 / 识别失败残留的原图，以及被删物品
    遗留的缩略图。只删确无引用的文件，安全可重复执行。
    """
    referenced = {i.image_path for i in db.query(Item).all() if i.image_path}
    item_ids = {i.id for i in db.query(Item).all()}

    removed_originals = 0
    for sub in config.UPLOAD_DIR.iterdir():
        if not sub.is_dir() or sub.name == "thumbs":
            continue
        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rel = f.relative_to(config.UPLOAD_DIR).as_posix()
                if rel not in referenced and f.exists():
                    _safe_unlink(f)
                    if not f.exists():
                        removed_originals += 1

    removed_thumbs = 0
    thumbs_dir = config.UPLOAD_DIR / "thumbs"
    if thumbs_dir.exists():
        for f in thumbs_dir.glob("item_*.jpg"):
            try:
                iid = int(f.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if iid not in item_ids and f.exists():
                _safe_unlink(f)
                if not f.exists():
                    removed_thumbs += 1
    return {"removed_originals": removed_originals,
            "removed_thumbs": removed_thumbs}
