"""
error_handler.py — 全局错误处理中间件

捕获 Agent 执行过程中的各种异常，返回友好的错误信息，
而不是把堆栈信息直接暴露给前端。
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import traceback
import os

from app.core.logger import get_logger

logger = get_logger("error_handler")


def register_error_handlers(app: FastAPI):
    """
    注册全局错误处理器到 FastAPI 应用。
    在 main.py 的 startup 之后调用。
    """

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        捕获所有未处理的异常。
        """
        logger.error("未处理异常: %s", exc, exc_info=True)

        # 返回友好错误信息给前端
        return JSONResponse(
            status_code=500,
            content={
                "error": "服务内部错误",
                "message": "Agent 执行失败，请稍后重试",
                "detail": str(exc) if os.getenv("DEBUG") else "内部错误，请联系管理员"  # 开发环境可以返回详细信息
            }
        )

    @app.exception_handler(TimeoutError)
    async def timeout_handler(request: Request, exc: TimeoutError):
        """处理超时错误"""
        return JSONResponse(
            status_code=504,
            content={
                "error": "请求超时",
                "message": "Agent 响应超时，请简化问题或稍后重试"
            }
        )
