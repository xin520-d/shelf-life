"""数据导出：CSV / XLSX（模块 10）。"""
import csv
import io
import datetime as dt
from typing import List

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from .date_parser import expiry_status

HEADERS = ["物品名称", "类别", "数量", "生产日期", "保质期(天)",
           "到期日期", "剩余天数", "状态", "检测置信度", "日期来源", "备注", "录入时间"]

_STATUS_LABEL = {"normal": "正常", "near": "临期", "expired": "已过期", "unknown": "未知"}


def _row(item, today: dt.date, near_days: int) -> List:
    status, remaining = expiry_status(item.expiry_date, today, near_days)
    return [
        item.name,
        item.category.name if item.category else "",
        f"{item.quantity}{item.unit}",
        item.production_date.strftime("%Y-%m-%d") if item.production_date else "",
        item.shelf_life_days if item.shelf_life_days is not None else "",
        item.expiry_date.strftime("%Y-%m-%d") if item.expiry_date else "",
        remaining if remaining is not None else "",
        _STATUS_LABEL.get(status, status),
        f"{item.confidence:.0%}" if item.confidence is not None else "",
        {"ocr": "OCR识别", "manual": "人工填写", "category_default": "类别默认"}.get(item.date_source, item.date_source),
        item.note or "",
        item.created_at.strftime("%Y-%m-%d %H:%M") if item.created_at else "",
    ]


def to_csv(items, today: dt.date, near_days: int) -> bytes:
    """CSV（utf-8-sig，Excel 直接打开不乱码）。"""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADERS)
    for it in items:
        w.writerow(_row(it, today, near_days))
    return buf.getvalue().encode("utf-8-sig")


def to_xlsx(items, today: dt.date, near_days: int) -> bytes:
    """XLSX：清单 + 汇总两个工作表，带临期着色。"""
    wb = Workbook()
    ws = wb.active
    ws.title = "保质期清单"

    header_fill = PatternFill("solid", fgColor="1F2937")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    thin = Side(style="thin", color="D1D5DB")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    fills = {
        "near": PatternFill("solid", fgColor="FEF3C7"),    # 琥珀
        "expired": PatternFill("solid", fgColor="FEE2E2"),  # 红
    }

    ws.append(HEADERS)
    for c in range(1, len(HEADERS) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill, cell.font = header_fill, header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border

    for it in items:
        ws.append(_row(it, today, near_days))
        r = ws.max_row
        status, _ = expiry_status(it.expiry_date, today, near_days)
        for c in range(1, len(HEADERS) + 1):
            cell = ws.cell(row=r, column=c)
            cell.border = border
            cell.alignment = Alignment(vertical="center")
            if status in fills:
                cell.fill = fills[status]

    widths = [22, 8, 8, 12, 10, 12, 10, 8, 10, 10, 24, 18]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"

    # 汇总表
    ws2 = wb.create_sheet("汇总")
    ws2.append(["指标", "数值"])
    for c in (1, 2):
        cell = ws2.cell(row=1, column=c)
        cell.fill, cell.font = header_fill, header_font
    stat = {"normal": 0, "near": 0, "expired": 0, "unknown": 0}
    by_cat: dict = {}
    for it in items:
        s, _ = expiry_status(it.expiry_date, today, near_days)
        stat[s] += 1
        cname = it.category.name if it.category else "未分类"
        by_cat[cname] = by_cat.get(cname, 0) + 1
    ws2.append(["导出时间", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    ws2.append(["物品总数", len(items)])
    ws2.append(["正常", stat["normal"]])
    ws2.append([f"临期(≤{near_days}天)", stat["near"]])
    ws2.append(["已过期", stat["expired"]])
    ws2.append(["日期未知", stat["unknown"]])
    ws2.append([])
    ws2.append(["按类别统计", ""])
    for k, v in sorted(by_cat.items(), key=lambda x: -x[1]):
        ws2.append([k, v])
    ws2.column_dimensions["A"].width = 20
    ws2.column_dimensions["B"].width = 14

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
