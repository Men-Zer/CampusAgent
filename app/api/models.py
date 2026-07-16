"""
models.py — 定义 API 的请求和响应数据模型

ChatRequest：前端发给后端的数据
ChatResponse：后端返回给前端的数据
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    """
    请求模型：前端发送的用户消息。
    示例：{"user_id": "user_001", "session_id": "session_001", "message": "我的IP是多少？"}
    """
    message: str = Field(..., max_length=4000)  # 用户的问题，最多 4000 字
    user_id: str = Field("default_user", max_length=64)  # 用户 ID（默认值，方便测试）
    session_id: str = Field("default_session", max_length=64)  # 会话 ID（默认值，方便测试）


class ChatResponse(BaseModel):
    """
    响应模型：后端返回的 Agent 回答。
    """
    answer: str
    sources: Optional[List[str]] = []
    context: Optional[str] = ""
    tool_calls: Optional[List[dict]] = []  # Agent 工具调用记录
