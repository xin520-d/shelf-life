"""下载 CLIP 零样本视觉识别模型（openai/clip-vit-base-patch32）并预热文本嵌入缓存。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import clip_cls

print("开始下载 CLIP 模型（约 600MB，首次较慢，支持断点续传）...", flush=True)
ok = clip_cls.download_model()
if not ok:
    print("下载失败", flush=True)
    sys.exit(1)
print("模型已就绪，预热文本嵌入缓存...", flush=True)
clip_cls.warmup()
print("CLIP 视觉识别模型安装完成", flush=True)
