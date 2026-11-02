"""Pydantic 请求/响应模型。"""
import datetime as dt
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------- 类别 ----------
class CategoryOut(BaseModel):
    id: int
    name: str
    color: str
    default_shelf_life_days: int
    is_builtin: bool
    item_count: int = 0

    class Config:
        from_attributes = True


class CategoryIn(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    color: str = "#6b7280"
    default_shelf_life_days: int = 365


# ---------- 识别草稿 ----------
class Candidate(BaseModel):
    name: str
    category_id: int
    category_name: str
    default_shelf_life_days: int


class DetectionDraft(BaseModel):
    """单次识别产生的一条物品草稿（前端审核后确认入库）。"""
    name: str
    raw_name: str = ""
    confidence: Optional[float] = None
    bbox: List[int] = []
    category_id: int
    category_name: str = ""
    quantity: int = 1
    unit: str = "件"

    production_date: Optional[dt.date] = None
    shelf_life_days: Optional[int] = None
    expiry_date: Optional[dt.date] = None
    date_text: str = ""
    date_source: str = "manual"
    date_note: str = ""                     # 机器建议的说明，如"未识别到日期，按类别默认"

    needs_review: bool = False
    candidates: List[Candidate] = []
    note: str = ""
    # 多物品切分区域（一张照片里摆着好几件东西时，每个区域各一条草稿）。
    # 这类草稿按"互不重叠的区域"生成，清洗时不能被"相邻同名"规则并回一条。
    region_split: bool = False


class PhotoResult(BaseModel):
    """一张照片的识别结果。"""
    filename: str
    image_url: str
    image_width: int = 0
    image_height: int = 0
    ocr_texts: List[str] = []               # 全文 OCR 文本（调试/透明度）
    drafts: List[DetectionDraft] = []
    message: str = ""
    clip_top: List[dict] = []               # CLIP 零样本视觉初判 Top5
    is_screenshot: bool = False             # 上传的是 App 界面截图


# ---------- 物品 ----------
class ItemIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    raw_name: str = ""
    category_id: int
    quantity: int = 1
    unit: str = "件"
    image_path: str = ""
    bbox: str = ""
    production_date: Optional[dt.date] = None
    shelf_life_days: Optional[int] = None
    expiry_date: Optional[dt.date] = None
    date_text: str = ""
    date_source: str = "manual"
    confidence: Optional[float] = None
    needs_review: bool = False
    note: str = ""


class ItemPatch(BaseModel):
    name: Optional[str] = None
    category_id: Optional[int] = None
    quantity: Optional[int] = None
    unit: Optional[str] = None
    production_date: Optional[dt.date] = None
    shelf_life_days: Optional[int] = None
    expiry_date: Optional[dt.date] = None
    date_source: Optional[str] = None
    note: Optional[str] = None
    needs_review: Optional[bool] = None


class ItemOut(BaseModel):
    id: int
    name: str
    raw_name: str
    category_id: int
    category_name: str
    category_color: str
    quantity: int
    unit: str
    image_path: str
    thumb_url: Optional[str] = None
    production_date: Optional[dt.date]
    shelf_life_days: Optional[int]
    expiry_date: Optional[dt.date]
    date_text: str
    date_source: str
    confidence: Optional[float]
    needs_review: bool
    note: str
    status: str
    remaining_days: Optional[int] = None
    expiry_status: str = "unknown"          # normal / near / expired / unknown
    created_at: Optional[dt.datetime]

    class Config:
        from_attributes = True


class BulkItemsIn(BaseModel):
    items: List[ItemIn]


# ---------- 统计 ----------
class StatsOut(BaseModel):
    total_active: int
    near_expiry: int
    expired: int
    normal: int
    due_7d: int
    due_30d: int
    unknown: int
    by_category: List[dict]
    expiry_timeline: List[dict]             # 未来 30 天到期趋势
    recent_items: List[ItemOut]


# ---------- 设置 ----------
class SettingIn(BaseModel):
    near_expiry_days: int = Field(ge=1, le=90)
    low_conf: float = Field(default=0.6, ge=0.0, le=1.0,
                            description="低于该值的识别结果进入待确认")


# ---------- 修正记忆 ----------
class CorrectionOut(BaseModel):
    id: int
    raw_name: str
    item_name: str
    category_name: str
    use_count: int
    last_used_at: Optional[dt.datetime]

    class Config:
        from_attributes = True
