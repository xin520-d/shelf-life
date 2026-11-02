"""日期解析引擎单测（纯逻辑，无需加载模型）。"""
from datetime import date

from app.services.date_parser import (
    DateParseResult,
    add_months,
    expiry_status,
    normalize_text,
    parse_date_text,
    suggest_expiry,
    _parse_date_token,
)


def test_normalize_text():
    # 全角转半角 + 空白压缩为单空格（必须保留空格，否则日期被粘成一串）
    assert normalize_text("２０２４　年") == "2024 年"
    assert normalize_text("  水果  罐头  ") == "水果 罐头"


def test_parse_date_token_formats():
    assert _parse_date_token("2024-01-15") == date(2024, 1, 15)
    assert _parse_date_token("2024.1.5") == date(2024, 1, 5)
    assert _parse_date_token("2024年1月15日") == date(2024, 1, 15)
    assert _parse_date_token("20240115") == date(2024, 1, 15)


def test_add_months_month_end_clamp():
    # 1月31 + 1个月 -> 2月末钳制（闰年 29 天）
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2023, 1, 31), 1) == date(2023, 2, 28)


def test_parse_production_plus_shelf():
    text = "生产日期：2024-01-15 保质期：12个月"
    r = parse_date_text(text)
    assert r.production_date == date(2024, 1, 15)
    assert r.expiry_date == date(2025, 1, 15)
    assert r.shelf_life_days == 366  # 跨闰年 2 月


def test_parse_explicit_expiry():
    r = parse_date_text("保质期至：2027.03.31")
    assert r.expiry_date == date(2027, 3, 31)


def test_parse_cn_duration():
    r = parse_date_text("生产日期 2023/06/01 保质期 三年")
    assert r.production_date == date(2023, 6, 1)
    assert r.expiry_date == date(2026, 6, 1)


def test_parse_dotmatrix():
    r = parse_date_text("PD: 2015 03 06 ED: 2016 03 06")
    assert r.production_date == date(2015, 3, 6)
    assert r.expiry_date == date(2016, 3, 6)


def test_parse_no_date_warns():
    r = parse_date_text("这是一包很好吃的零食")
    assert not r.has_any
    assert r.warnings  # 无任何日期应给定位提示


def test_expiry_status():
    today = date(2024, 6, 1)
    assert expiry_status(date(2024, 6, 10), today, 7) == ("normal", 9)
    assert expiry_status(date(2024, 6, 5), today, 7) == ("near", 4)
    assert expiry_status(date(2024, 5, 20), today, 7) == ("expired", -12)
    assert expiry_status(None, today, 7) == ("unknown", None)


def test_suggest_expiry_explicit():
    prod, days, exp, notes = suggest_expiry(
        date(2024, 1, 1), 365, 365, date(2025, 1, 1), date(2024, 6, 1))
    assert exp == date(2025, 1, 1)
