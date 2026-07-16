"""
main.py — 应用启动入口

职责：
1. 创建 FastAPI 应用
2. 注册路由（/chat、/chat/stream）
3. 注册错误处理中间件
4. 启动时初始化 Agent 服务
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.services.react_agent import ReactAgent
from app.services.memory_service import MemoryService
from app.api.routes import router
import app.api.routes as routes_module
from app.api.error_handler import register_error_handlers
import os
import sys
from app.core.logger import init_logging, get_logger

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理。
    startup 阶段初始化服务，shutdown 阶段清理资源。
    """
    # === startup ===
    init_logging()
    logger.info("HarmonyCampus 启动中...")

    try:
        logger.info("正在初始化 Agent 服务...")
        routes_module.agent_service = ReactAgent()
        logger.info("Agent 服务初始化完成")

        logger.info("正在初始化记忆服务...")
        routes_module.memory_service = MemoryService()
        logger.info("记忆服务初始化完成")

        logger.info("HarmonyCampus 启动完成")
    except Exception as e:
        logger.error("启动失败: %s", e, exc_info=True)
        sys.exit(1)

    yield  # 应用运行中

    # === shutdown ===
    logger.info("HarmonyCampus 正在关闭...")
    try:
        from app.api.routes import _executor
        _executor.shutdown(wait=False, cancel_futures=True)
        logger.info("线程池已关闭")
    except Exception:
        pass


# 创建 FastAPI 应用实例
app = FastAPI(
    title="HarmonyCampus",
    description="校园 AI 智能助手（Agent + RAG）",
    lifespan=lifespan,
)

# 注册错误处理中间件
register_error_handlers(app)

# 注册路由：/chat、/chat/stream
app.include_router(router)


@app.get("/api/health")
def health():
    """健康检查接口"""
    return {
        "message": "HarmonyCampus 已启动",
        "endpoints": {
            "非流式问答": "POST /chat",
            "流式问答": "POST /chat/stream",
        }
    }


# 挂载静态文件（前端页面），必须放在所有 API 路由之后
# 否则 StaticFiles 会拦截所有请求
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=os.path.join(_BASE_DIR, "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    _host = os.getenv("HOST", "127.0.0.1")
    _port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=_host, port=_port)
