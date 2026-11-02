# shelf-life 后端 (FastAPI)

保质期 / 食品识别后端服务：OCR（RapidOCR）+ YOLO 检测 + CLIP 零样本分类。

## 目录结构
- `app/` FastAPI 应用（routers + services）
- `run.py` 启动入口（`python run.py`，使用 uvicorn）
- `requirements.txt` 依赖
- `scripts/` 辅助脚本（含 `download_clip.py` 下载 CLIP 模型）
- `tests/` 测试
- `data/` 测试用样例（csv/xlsx/图片）
- `models/`、`uploads/` 为运行时目录，权重与上传文件**不入库**（见 `.gitignore`）

## 环境搭建
```bash
pip install -r requirements.txt
python scripts/download_clip.py      # 下载 CLIP 模型（首次较慢，约 600MB）
python run.py                        # 启动服务（默认 http://127.0.0.1:8000）
pytest                               # 运行测试
```
YOLO 权重在首次推理时由 ultralytics 自动下载，无需手动处理。

## 说明
模型权重（如 `models/clip_l14/pytorch_model.bin`）体积超过 1.5GB，超出 GitHub 单文件限制，
故不纳入版本库，请按上述脚本在部署环境自行下载。
