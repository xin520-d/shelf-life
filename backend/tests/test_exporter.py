"""导出模块单测（用 fake item，不触达数据库）。"""
from datetime import date, datetime
from types import SimpleNamespace

from app.services import exporter


def _fake_item(**kw):
    item = SimpleNamespace(
        name="测试薯片",
        category=SimpleNamespace(name="食品"),
        quantity=2,
        unit="包",
        production_date=date(2024, 1, 1),
        shelf_life_days=365,
        expiry_date=date(2025, 1, 1),
        confidence=0.8,
        date_source="ocr",
        note="备注信息",
        created_at=datetime(2024, 1, 2, 10, 0),
    )
    for k, v in kw.items():
        setattr(item, k, v)
    return item


def test_headers_count():
    # 位置模块删除后应为 12 列
    assert len(exporter.HEADERS) == 12


def test_row_columns():
    row = exporter._row(_fake_item(), date(2024, 6, 1), 7)
    assert len(row) == 12
    assert row[0] == "测试薯片"
    assert row[7] == "正常"  # 状态列


def test_to_csv():
    data = exporter.to_csv([_fake_item()], date(2024, 6, 1), 7)
    assert isinstance(data, bytes)
    text = data.decode("utf-8-sig")  # CSV 用 utf-8-sig，Excel 不乱码
    assert "物品名称" in text
    assert "测试薯片" in text


def test_to_xlsx():
    data = exporter.to_xlsx([_fake_item()], date(2024, 6, 1), 7)
    assert isinstance(data, bytes)
    assert len(data) > 0
