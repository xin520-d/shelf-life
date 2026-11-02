"""按新审核判定规则重算库内物品的 needs_review 标记。

新规则（与 pipeline._decide_review 一致）：仅当存在"硬"不确定信号时才待确认：
  - 置信度非空且 < 阈值 -> 待确认（识别把握不足）
  - 无日期（生产/到期皆空） -> 待确认（无法估算到期）
  - 日期明显可疑（已过期 或 生产日期早于 N 年） -> 待确认
  - 其他（含置信度为 None 但已读到期日且日期正常） -> 免审，AI 推理说明仍保留展示

策略：对全部物品重算，双向翻转，使库内标记与当前规则完全一致。

用法：
  python _refresh_review_flags.py          # dry-run，仅统计
  python _refresh_review_flags.py --apply  # 真正写库
"""
import sys

from sqlalchemy import inspect, text

from app.database import engine, SessionLocal
from app.models import Item, Setting
from app.services import pipeline
from app.services.pipeline import _decide_review, SUSPECT_OLD_YEARS


def migrate_settings_column():
    """兼容旧库：settings 表缺少 low_conf 列则补加（幂等）。"""
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns("settings")]
    if "low_conf" not in cols:
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE settings ADD COLUMN low_conf FLOAT DEFAULT 0.6"))


def get_low_conf(db):
    s = db.get(Setting, 1)
    if s and s.low_conf is not None:
        return float(s.low_conf)
    return 0.6


def _fake_parsed(it):
    """用 Item 的日期字段拼一个最小 parsed 对象，喂给 _decide_review。"""
    class _P:
        pass
    p = _P()
    p.production_date = it.production_date
    p.expiry_date = it.expiry_date
    p.warnings = []
    p.has_any = bool(it.production_date or it.expiry_date)
    return p


def main(dry_run: bool = True):
    migrate_settings_column()
    db = SessionLocal()
    low_conf = get_low_conf(db)
    items = db.query(Item).all()
    total = len(items)
    rev_before = sum(1 for it in items if it.needs_review)

    to_review, to_clear = 0, 0
    for it in items:
        should = bool(_decide_review(it.confidence, _fake_parsed(it), low_conf))
        if should and not it.needs_review:
            if not dry_run:
                it.needs_review = True
            to_review += 1
            if dry_run:
                print(f"  [新判待确认] id={it.id} name={it.name!r} "
                      f"prod={it.production_date} exp={it.expiry_date}")
        elif (not should) and it.needs_review:
            if not dry_run:
                it.needs_review = False
            to_clear += 1
            if dry_run:
                print(f"  [新判免审] id={it.id} name={it.name!r}")

    if not dry_run:
        db.commit()

    print(f"\n库内物品总数: {total} | 重算前待确认: {rev_before}")
    print(f"阈值 low_conf = {low_conf} | 旧货阈值 = {SUSPECT_OLD_YEARS} 年")
    if dry_run:
        print("=== DRY RUN（未写库）===")
    print(f"本次新判待确认(+): {to_review}")
    print(f"本次新判免审(-): {to_clear}")
    print(f"重算后待确认: {rev_before + to_review - to_clear}")
    db.close()


if __name__ == "__main__":
    dry = "--apply" not in sys.argv
    main(dry_run=dry)
