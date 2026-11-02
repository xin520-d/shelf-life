"""审核判定 _decide_review 单测（纯逻辑，无需模型）。"""
from datetime import date

from app.services.date_parser import DateParseResult
from app.services.pipeline import SUSPECT_OLD_YEARS, _decide_review


def _parsed(prod=None, exp=None, warnings=None, has_any=None):
    if warnings is None:
        warnings = []
    if has_any is None:
        has_any = bool(prod or exp)
    return DateParseResult(production_date=prod, expiry_date=exp, warnings=warnings)


def test_high_conf_with_date_auto():
    # 高置信 + 有日期(未过期、不旧) + 无告警 -> 自动入库（不卡待确认）
    p = _parsed(prod=date(2026, 1, 1), exp=date(2027, 1, 1))
    assert _decide_review(0.9, p, 0.6) is False


def test_low_conf_blocks():
    p = _parsed(prod=date(2026, 1, 1))
    assert _decide_review(0.3, p, 0.6) is True


def test_conf_none_not_low():
    # OCR 路径 conf=None，不应触发低置信；日期正常(未过期、不旧)则不卡
    p = _parsed(prod=date(2026, 1, 1))
    assert _decide_review(None, p, 0.6) is False


def test_no_date_blocks():
    assert _decide_review(0.9, None, 0.6) is True
    assert _decide_review(0.9, _parsed(), 0.6) is True


def test_warning_blocks():
    p = _parsed(prod=date(2024, 1, 1), warnings=["生产日期晚于今天，请核对"])
    assert _decide_review(0.9, p, 0.6) is True


def test_old_production_blocks():
    old = date(date.today().year - (SUSPECT_OLD_YEARS + 1), 1, 1)
    p = _parsed(prod=old)
    assert _decide_review(0.9, p, 0.6) is True


def test_expired_blocks():
    exp = date(date.today().year - 1, 1, 1)
    p = _parsed(prod=date(date.today().year - 2, 1, 1), exp=exp)
    assert _decide_review(0.9, p, 0.6) is True


def test_prod_2024_not_suspect():
    # 放宽 SUSPECT_OLD_YEARS 到 5 后：2024 年生产的正常未过期物品
    # （在 2026 年）不应因"生产日期早"被强制进待确认。
    p = _parsed(prod=date(2024, 6, 1), exp=date(2027, 6, 1))
    assert _decide_review(0.9, p, 0.6) is False
