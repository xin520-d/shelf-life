"""调试：对用户上传的图片跑 OCR + 日期解析，打印中间结果。"""
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import ocr
from app.services.date_parser import parse_date_text, suggest_expiry, normalize_text

img_path = Path(sys.argv[1])
pil = Image.open(img_path)
pil.load()
print(f"图片: {img_path.name}  尺寸: {pil.size}")

items = ocr.ocr(pil)
print(f"\n=== OCR 共 {len(items)} 行 ===")
for it in sorted(items, key=lambda t: (sum(p[1] for p in t["box"]) / 4, sum(p[0] for p in t["box"]) / 4)):
    cy = sum(p[1] for p in it["box"]) / 4
    cx = sum(p[0] for p in it["box"]) / 4
    print(f"  y={cy:7.1f} x={cx:7.1f} score={it['score']:.2f}  {it['text']!r}")

full = " ".join(t["text"] for t in items)
print("\n=== 拼接全文 ===")
print(normalize_text(full))

print("\n=== 日期解析 ===")
parsed = parse_date_text(full)
print(f"production_date = {parsed.production_date}")
print(f"shelf_life_days = {parsed.shelf_life_days}")
print(f"expiry_date     = {parsed.expiry_date}")
print(f"matched_texts   = {parsed.matched_texts}")
print(f"warnings        = {parsed.warnings}")

prod, days, expiry, notes = suggest_expiry(
    parsed.production_date, parsed.shelf_life_days, 540,
    expiry_date=parsed.expiry_date)
print(f"\nsuggest -> prod={prod} days={days} expiry={expiry} notes={notes}")
