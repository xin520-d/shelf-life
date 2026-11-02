"""ORM 数据模型：类别、位置、物品、用户修正记忆、系统设置。"""
import datetime as dt
from sqlalchemy import (Column, Integer, String, Float, Boolean, Date,
                        DateTime, ForeignKey, Text, Enum as SAEnum, UniqueConstraint)
from sqlalchemy.orm import relationship
from .database import Base


class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    color = Column(String(20), default="#6b7280")
    default_shelf_life_days = Column(Integer, default=365)
    is_builtin = Column(Boolean, default=False)


class Location(Base):
    __tablename__ = "locations"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    icon = Column(String(20), default="📦")
    is_builtin = Column(Boolean, default=False)


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)                 # 物品名称（用户可改）
    raw_name = Column(String(100), default="")                 # 模型原始输出
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=True)
    quantity = Column(Integer, default=1)
    unit = Column(String(20), default="件")

    image_path = Column(String(500), default="")               # uploads 内相对路径
    bbox = Column(Text, default="")                            # JSON: [x1,y1,x2,y2]

    category = relationship("Category", lazy="joined")
    location = relationship("Location", lazy="joined")

    production_date = Column(Date, nullable=True)              # 生产日期
    shelf_life_days = Column(Integer, nullable=True)           # 保质期天数
    expiry_date = Column(Date, nullable=True)                  # 到期日期（归一化结果）
    date_text = Column(String(200), default="")                # OCR 原文
    date_source = Column(String(20), default="manual")         # ocr / manual / category_default

    confidence = Column(Float, nullable=True)                  # 检测置信度
    needs_review = Column(Boolean, default=False)              # 需人工确认
    note = Column(String(500), default="")

    status = Column(String(20), default="active")              # active / consumed / discarded
    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
                        onupdate=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    finished_at = Column(DateTime, nullable=True)              # 消耗/丢弃时间


class CorrectionMemory(Base):
    """类别映射与用户修正记忆：模型输出 raw_name -> 用户选定的类别。"""
    __tablename__ = "correction_memory"
    __table_args__ = (UniqueConstraint("raw_name", name="uq_raw_name"),)
    id = Column(Integer, primary_key=True, index=True)
    raw_name = Column(String(100), nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    item_name = Column(String(100), default="")                # 用户常用的规范名
    use_count = Column(Integer, default=1)
    last_used_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))


class Setting(Base):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True)
    near_expiry_days = Column(Integer, default=7)
    low_conf = Column(Float, default=0.6)   # 低于该值的识别结果进入"待确认"
