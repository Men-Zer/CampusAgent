"""
logger.py — 统一日志系统

提供集中式日志配置和 get_logger() 工具函数：
- 控制台 Rich 美化输出 + 文件双输出
- 请求级 request_id 追踪（通过 ContextVar）
- 按天轮转（TimedRotatingFileHandler），自动清理过期日志
"""

import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler
from contextvars import ContextVar

from rich.console import Console
from rich.logging import RichHandler

# 日志目录
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)

# 文件日志格式（精确到毫秒，适合排查问题）
FILE_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-5s | %(req_id)s | "
    "%(name)-20s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 用于追踪当前请求的 contextvar（每个请求独立）
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestFilter(logging.Filter):
    """注入当前请求ID到日志记录"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.req_id = request_id_var.get()
        return True


# 初始化标志
_initialized = False


def init_logging(level: int = logging.DEBUG, keep_days: int = 30) -> None:
    """初始化日志系统，在 main.py 启动时调用一次。"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    os.makedirs(LOG_DIR, exist_ok=True)

    # 文件 handler：按天轮转，保留 keep_days 天
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, "agenticrag.log"),
        when="midnight",
        interval=1,
        backupCount=keep_days,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT, DATE_FORMAT))
    file_handler.addFilter(RequestFilter())

    # 控制台 handler：Rich 美化输出
    console = Console()
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
    )
    rich_handler.setLevel(logging.INFO)
    rich_handler.addFilter(RequestFilter())

    # 根 logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(rich_handler)

    # 抑制第三方库的 DEBUG 噪音
    for lib in ["chromadb", "urllib3", "httpx", "httpcore", "openai", "tiktoken"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.getLogger("app").info(
        "日志系统初始化完成 | 文件: %s",
        os.path.join(LOG_DIR, "agenticrag.log"),
    )


def get_logger(name: str) -> logging.Logger:
    """获取带模块名的 logger。

    Args:
        name: logger 名称，通常传 __name__ 或模块短名

    Returns:
        配置好的 Logger 实例
    """
    return logging.getLogger(name)
