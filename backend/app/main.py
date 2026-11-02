"""FastAPI 应用入口。"""
import datetime as dt
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config
from .config import BASE_DIR
from .database import get_db, engine, SessionLocal, Base
from .models import Category, Location, Setting
from .routers import recognize, items, meta, stats, export

FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"


def seed_data():
    """首次启动时写入内置类别、存放位置与默认设置。"""
    db = SessionLocal()
    try:
        if db.query(Category).count() == 0:
            for name, color, days in config.BUILTIN_CATEGORIES:
                db.add(Category(name=name, color=color,
                                default_shelf_life_days=days, is_builtin=True))
        if db.query(Location).count() == 0:
            for name, icon in [("冰箱冷藏室", "🧊"), ("冰箱冷冻室", "❄️"),
                               ("厨房储物架", "🍽️"), ("阳台柜第二格", "📦"),
                               ("卧室储物柜", "🚪"), ("药箱", "💊")]:
                db.add(Location(name=name, icon=icon, is_builtin=True))
        if db.get(Setting, 1) is None:
            db.add(Setting(id=1, near_expiry_days=config.NEAR_EXPIRY_DAYS,
                           low_conf=config.LOW_CONF))
        db.commit()
    finally:
        db.close()


def _migrate_settings_column():
    """兼容旧库：settings 表若缺少 low_conf 列则补加（SQLite 不支持 ALTER 改列，仅加列）。"""
    from sqlalchemy import inspect, text
    try:
        insp = inspect(engine)
        cols = [c["name"] for c in insp.get_columns("settings")]
        if "low_conf" not in cols:
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE settings ADD COLUMN low_conf FLOAT DEFAULT 0.6"))
    except Exception:
        pass


def create_app() -> FastAPI:
    Base.metadata.create_all(bind=engine)
    _migrate_settings_column()
    seed_data()

    app = FastAPI(title="物品保质期识别与提醒", version="1.0.0")

    logger = logging.getLogger("bzq.api")

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422,
                            content={"detail": "请求参数校验失败",
                                     "errors": jsonable_encoder(exc.errors())})

    @app.exception_handler(Exception)
    async def _unhandled_exc(request: Request, exc: Exception):
        logger.exception("未捕获异常 %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "服务器内部错误，请稍后重试或联系管理员"})

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- API 路由（须在静态托管之前注册） ----------
    app.include_router(recognize.router)
    app.include_router(items.router)
    app.include_router(meta.router)
    app.include_router(stats.router)
    app.include_router(export.router)

    _yolo_load_state = {"checked": False, "ok": False}

    def _yolo_weights_ok() -> bool:
        """YOLO 权重是否真实可加载（存在≠可用，曾因截断文件误报 healthy）。

        结果按进程缓存一次：model_ready() 内部是单例，重复调用不重复加载。
        """
        weights = config.MODEL_DIR / config.YOLO_MODEL
        if not weights.exists():
            return False
        if not _yolo_load_state["checked"]:
            try:
                from .services.detector import model_ready
                _yolo_load_state["ok"] = model_ready()
            except Exception:
                _yolo_load_state["ok"] = False
            _yolo_load_state["checked"] = True
        return _yolo_load_state["ok"]

    @app.get("/api/health")
    def health(db: Session = Depends(get_db)):
        yolo_ok = _yolo_weights_ok()
        try:
            from .services import ocr as ocr_service
            ocr_ok = ocr_service.ocr_ready()
        except Exception:
            ocr_ok = False
        count = db.query(Category).count()
        return {
            "status": "ok",
            "yolo_weights": yolo_ok,
            "ocr_engine": ocr_ok,
            "seeded": count > 0,
            "time": dt.datetime.now().isoformat(timespec="seconds"),
        }

    # ---------- 上传照片 ----------
    app.mount("/uploads", StaticFiles(directory=str(config.UPLOAD_DIR)), name="uploads")

    # ---------- 托管前端构建产物（frontend/dist）：单服务运行完整应用 ----------
    if FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            """SPA 回退：未匹配的前端路由一律返回 index.html。"""
            index = FRONTEND_DIST / "index.html"
            resp = FileResponse(index)
            # 防止浏览器/内嵌预览长期缓存旧 HTML，导致前端更新后看不到
            resp.headers["Cache-Control"] = "no-cache"
            return resp

    # ---------- 启动预热：把三个模型（YOLO / RapidOCR / CLIP）的加载与
    #   首次推理移到服务启动阶段，消除"用户第一次上传才现加载"的 ~30 秒卡顿。
    #   后台守护线程执行，不阻塞服务启动；与 health 端点的懒加载共用同一把锁，
    #   不会重复加载。模型未下载（CLIP 可选）时优雅跳过，不影响主流程。
    import threading

    def _warmup_models() -> None:
        try:
            from .services.detector import model_ready
            model_ready()
        except Exception:
            pass
        try:
            from .services import ocr as _ocr_svc
            _ocr_svc.ocr_ready()
        except Exception:
            pass
        try:
            from .services import clip_cls as _clip
            if _clip.available():
                _clip.warmup()
        except Exception:
            pass

    threading.Thread(target=_warmup_models, daemon=True).start()

    return app


app = create_app()
