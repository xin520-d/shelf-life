"""启动入口：python run.py（host/port 可在 app/config.py 或环境变量配置）。"""
import uvicorn

from app import config

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=config.BIND_HOST, port=config.PORT,
                reload=False)
