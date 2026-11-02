"""生成模拟"冰箱/储物架照片"用于端到端测试（ PIL 绘制包装 + 中文日期文本）。"""
from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 700
img = Image.new("RGB", (W, H), (228, 231, 236))  # 冰箱内部灰白背景
d = ImageDraw.Draw(img)
font_big = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 30)
font_mid = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
font_sm = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)

# 隔板
d.rectangle([0, 460, W, 470], fill=(180, 185, 190))

# 1. 牛奶盒（纸盒）
d.rectangle([60, 180, 300, 460], fill=(245, 245, 250), outline=(120, 120, 130), width=3)
d.polygon([(60, 180), (110, 130), (350, 130), (300, 180)], fill=(235, 240, 245), outline=(120, 120, 130))
d.text((95, 220), "纯牛奶", font=font_big, fill=(30, 60, 120))
d.text((80, 290), "净含量 1L", font=font_sm, fill=(80, 80, 80))
d.text((72, 340), "生产日期:2026-09-20", font=font_sm, fill=(20, 20, 20))
d.text((72, 375), "保质期:6个月", font=font_sm, fill=(20, 20, 20))

# 2. 瓶装水（瓶子轮廓）
d.rounded_rectangle([360, 240, 480, 460], radius=30, fill=(120, 190, 230), outline=(60, 120, 160), width=3)
d.rectangle([395, 190, 445, 245], fill=(100, 170, 210), outline=(60, 120, 160), width=3)
d.rectangle([375, 175, 465, 195], fill=(50, 100, 140))
d.text((372, 300), "矿泉水", font=font_mid, fill=(255, 255, 255))
d.text((368, 340), "保质期至", font=font_sm, fill=(255, 255, 255))
d.text((368, 370), "2027.05", font=font_sm, fill=(255, 255, 255))

# 3. 药品盒
d.rectangle([540, 240, 760, 460], fill=(250, 250, 250), outline=(160, 60, 60), width=3)
d.text((585, 275), "感冒灵颗粒", font=font_big, fill=(160, 40, 40))
d.text((560, 345), "生产日期 2025-01-10", font=font_sm, fill=(20, 20, 20))
d.text((560, 380), "有效期 24 个月", font=font_sm, fill=(20, 20, 20))

# 4. 面包袋（右上，已过期场景）
d.rounded_rectangle([790, 280, 960, 460], radius=20, fill=(240, 220, 180), outline=(150, 120, 60), width=3)
d.text((820, 320), "全麦面包", font=font_mid, fill=(90, 60, 20))
d.text((805, 370), "生产日期:", font=font_sm, fill=(60, 40, 10))
d.text((805, 398), "2026-09-28", font=font_sm, fill=(60, 40, 10))

# 底部：鸡蛋盒 + 酸奶
d.rectangle([60, 500, 420, 660], fill=(230, 220, 200), outline=(140, 120, 90), width=3)
d.text((180, 530), "鸡蛋 30枚", font=font_mid, fill=(80, 60, 30))
d.text((90, 580), "生产日期:2026-10-05 保质期:30天", font=font_sm, fill=(60, 40, 10))

d.rectangle([470, 510, 620, 660], fill=(250, 245, 240), outline=(200, 190, 180), width=3)
d.text((495, 545), "酸奶", font=font_mid, fill=(70, 60, 50))
d.text((480, 590), "生产日期:", font=font_sm, fill=(60, 40, 10))
d.text((480, 618), "2026-10-08", font=font_sm, fill=(60, 40, 10))

img.save("E:/保质期/bzq/backend/data/test_fridge.jpg", quality=92)
print("saved test_fridge.jpg", img.size)
