"""物品列表有效期筛选的单测（临期/已过期标签页的正确性）。

背景 bug：`expiring_within=N` 只判断 `remaining_days <= N`，已过期物品剩余
天数为负，也被判为「N 天内到期」，于是「临期」标签页里塞满了「已过期 2972 天」
的物品（2026-10-21 用户反馈）。
"""
import datetime as dt
from types import SimpleNamespace

from app.routers.items import _match_expiry_filter


def _o(remaining, state, status="active"):
    return SimpleNamespace(remaining_days=remaining, expiry_status=state, status=status)


def test_expiring_within_excludes_expired():
    # 已过期（剩余为负）不能算作「N 天内到期」
    assert not _match_expiry_filter(_o(-1, "expired"), 7, None)
    assert not _match_expiry_filter(_o(-2972, "expired"), 7, None)
    # 未过期且在窗口内 / 边界当天
    assert _match_expiry_filter(_o(0, "near"), 7, None)
    assert _match_expiry_filter(_o(7, "near"), 7, None)
    # 窗口外
    assert not _match_expiry_filter(_o(8, "normal"), 7, None)
    # 日期未知
    assert not _match_expiry_filter(_o(None, "unknown"), 7, None)


def test_status_filter_near_and_expired_are_disjoint():
    near = _o(3, "near")
    expired = _o(-30, "expired")
    assert _match_expiry_filter(near, None, "near")
    assert not _match_expiry_filter(expired, None, "near")
    assert _match_expiry_filter(expired, None, "expired")
    assert not _match_expiry_filter(near, None, "expired")


def test_finished_items_not_touched_by_window():
    # 已消耗/丢弃的物品不参与「N 天内」窗口过滤（供已消耗标签页查看）
    assert _match_expiry_filter(_o(-100, "expired", status="consumed"), 7, None)


def test_real_dates_regression():
    """用真实日期再走一遍 expiry_status，防阈值判定漂移。"""
    from app.services.date_parser import expiry_status

    today = dt.date(2026, 10, 21)
    expired_state, expired_rem = expiry_status(dt.date(2018, 9, 1), today, 7)
    assert expired_state == "expired" and expired_rem < 0
    assert not _match_expiry_filter(
        _o(expired_rem, expired_state), 7, None)

    near_state, near_rem = expiry_status(dt.date(2026, 10, 24), today, 7)
    assert near_state == "near" and near_rem == 3
    assert _match_expiry_filter(_o(near_rem, near_state), 7, None)
    assert _match_expiry_filter(_o(near_rem, near_state), None, "near")
