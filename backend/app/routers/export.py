"""导出路由：CSV / XLSX（模块 10）。"""
import datetime as dt
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Item
from ..services import exporter
from .items import _get_near_days

router = APIRouter(prefix="/api/export", tags=["export"])


def _query_items(db: Session, status: str, category_id: Optional[int],
                 location_id: Optional[int], q: Optional[str]):
    query = db.query(Item)
    if status != "all":
        query = query.filter(Item.status == status)
    if category_id:
        query = query.filter(Item.category_id == category_id)
    if location_id:
        query = query.filter(Item.location_id == location_id)
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Item.name.like(like), Item.note.like(like)))
    return query.all()


@router.get("/csv")
def export_csv(status: str = Query("active"), category_id: Optional[int] = None,
               location_id: Optional[int] = None, q: Optional[str] = None,
               db: Session = Depends(get_db)):
    items = _query_items(db, status, category_id, location_id, q)
    data = exporter.to_csv(items, dt.date.today(), _get_near_days(db))
    fname = f"保质期清单_{dt.datetime.now():%Y%m%d_%H%M}.csv"
    return StreamingResponse(
        iter([data]), media_type="text/csv; charset=utf-8-sig",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(fname)}"})


@router.get("/xlsx")
def export_xlsx(status: str = Query("active"), category_id: Optional[int] = None,
                location_id: Optional[int] = None, q: Optional[str] = None,
                db: Session = Depends(get_db)):
    items = _query_items(db, status, category_id, location_id, q)
    data = exporter.to_xlsx(items, dt.date.today(), _get_near_days(db))
    fname = f"保质期清单_{dt.datetime.now():%Y%m%d_%H%M}.xlsx"
    return StreamingResponse(
        iter([data]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(fname)}"})
