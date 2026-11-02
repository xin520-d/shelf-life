"""pytest 配置：把 backend 目录加入 sys.path，使测试能 `import app`。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
