"""统计看板路由（模块 4 的看板呈现）。"""
import datetime as dt
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Item, Setting
from ..schemas import StatsOut
from ..services.date_parser import expiry_status
from .. import config
from .items import _to_out, _get_near_days

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("", response_model=StatsOut)
def stats(db: Session = Depends(get_db)):
    today = dt.date.today()
    near = _get_near_days(db)
    items = db.query(Item).filter(Item.status == "active").all()

    stat = {"normal": 0, "near": 0, "expired": 0, "unknown": 0}
    by_cat: dict = {}
    timeline = [{"date": (today + dt.timedelta(days=i)).isoformat(),
                 "count": 0} for i in range(31)]
    timeline_map = {t["date"]: t for t in timeline}

    due_7d = due_30d = 0
    outs: List = []
    for it in items:
        st, remaining = expiry_status(it.expiry_date, today, near)
        stat[st] += 1
        if st == "expired" or (remaining is not None and 0 <= remaining <= 7):
            due_7d += 1
        if st == "expired" or (remaining is not None and 0 <= remaining <= 30):
            due_30d += 1
        if it.expiry_date:
            key = it.expiry_date.isoformat()
            if key in timeline_map:
                timeline_map[key]["count"] += 1
        cname = it.category.name if it.category else "未分类"
        ccolor = it.category.color if it.category else "#6b7280"
        if cname not in by_cat:
            by_cat[cname] = {"name": cname, "color": ccolor, "count": 0}
        by_cat[cname]["count"] += 1
        outs.append(_to_out(it, today, near))

    # 最紧急的 8 条
    outs.sort(key=lambda o: (o.remaining_days is None,
                             o.remaining_days if o.remaining_days is not None else 0))
    recent = outs[:8]

    return StatsOut(
        total_active=len(items),
        near_expiry=stat["near"],
        expired=stat["expired"],
        normal=stat["normal"],
        unknown=stat["unknown"],
        due_7d=due_7d,
        due_30d=due_30d,
        by_category=sorted(by_cat.values(), key=lambda x: -x["count"]),
        expiry_timeline=timeline,
        recent_items=recent,
    )
