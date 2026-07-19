"""
routes.py — API 路由定义

包含两个端点：
1. POST /chat       — 非流式，等 Agent 全部跑完再一次性返回
2. POST /chat/stream — 流式，逐 token 返回（像打字机效果）
"""

import asyncio
import re
import json as _json
import os
import time
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from app.api.models import ChatRequest, ChatResponse
from app.services.react_agent import ReactAgent
from app.services.memory_service import MemoryService
from app.core.logger import get_logger, request_id_var

logger = get_logger("routes")

router = APIRouter()

# Agent 和记忆服务实例，在 main.py 启动时注入
agent_service: ReactAgent = None
memory_service: MemoryService = None

# 线程池：把同步阻塞的 LLM 调用丢到独立线程，避免卡住 FastAPI 的事件循环
_executor = ThreadPoolExecutor(max_workers=4)


def extract_sources(result: dict) -> list:
    """
    从 Agent 执行结果中提取来源信息。
    遍历所有 ToolMessage，收集工具调用记录作为来源。
    """
    sources = []
    messages = result.get("messages", [])
    for msg in messages:
        # ToolMessage 是工具执行后的返回消息，name 字段是工具名
        if hasattr(msg, "name") and msg.name:
            tool_name = msg.name
            if tool_name == "search_knowledge_base":
                content = msg.content if hasattr(msg, "content") else ""
                found = re.findall(r"来自《(.+?)》", content)
                sources.extend(found)
            elif tool_name == "web_search":
                sources.append("互联网搜索")
    # dict.fromkeys() 去重（保持顺序）
    return list(dict.fromkeys(sources))


@router.post("/chat", response_model=ChatResponse)
async def chat(item: ChatRequest):
    """
    非流式问答接口。等 Agent 完整跑完后一次性返回。

    注意：Agent 内部调 Ollama 是同步阻塞的，
    必须用 run_in_executor 丢到线程池，否则会卡住 FastAPI 的 async 事件循环。
    """
    rid = (item.user_id[:8] if len(item.user_id) >= 8 else item.user_id) + "-" + _uuid.uuid4().hex[:6]
    request_id_var.set(rid)
    t0 = time.time()
    logger.info("━" * 60)
    logger.info("📨 [请求] user=%-12s session=%-12s q=%s",
                item.user_id[:8], item.session_id[:8], len(item.message))

    try:
        # 设置当前 user_id，工具内部通过 contextvar 读取
        from app.tools.agent_tools import _current_user_id
        _current_user_id.set(item.user_id)

        # 1. 获取记忆上下文
        memory_context = memory_service.build_memory_context(
            user_id=item.user_id,
            session_id=item.session_id,
            question=item.message,
        )

        # 组装发送给 LLM 的完整上下文
        from app.services.react_agent import SYSTEM_PROMPT
        full_prompt = SYSTEM_PROMPT.replace("{memory_context}", memory_context)
        full_context = f"{full_prompt}\n\n[用户消息] {item.message}"

        # 2. 调用 Agent（带回工具调用信息）
        loop = asyncio.get_running_loop()
        answer, tool_calls, result = await loop.run_in_executor(
            _executor,
            lambda: agent_service.get_answer_with_tools(item.message, memory_context, item.user_id),
        )

        # 后处理：模型输出的脏符号归一化
        if answer:
            answer = answer.replace('\\\\n', '\n')
            answer = re.sub(r'^#{2,3}\s+(.+)$', r'\n\n## \1\n\n', answer, flags=re.MULTILINE)
            answer = re.sub(r'\n{3,}', '\n\n', answer).strip()

        # 从 Agent 执行结果中提取来源（启用 extract_sources，废弃原来匹配不到的正则）
        # extract_sources 会遍历 ToolMessage，从 search_knowledge_base 的输出里
        # 提 "来自《xxx》"，从 web_search 里加 "互联网搜索"
        sources = extract_sources(result)

        # 保存记忆
        memory_service.after_response(
            user_id=item.user_id,
            session_id=item.session_id,
            user_msg=item.message,
            bot_reply=answer,
        )

        logger.info("✅ [完成]  %.2fs  reply=%-5d chars", time.time() - t0, len(answer) if answer else 0)
        logger.info("━" * 60)
        return ChatResponse(answer=answer, sources=sources, context=full_context, tool_calls=tool_calls)

    except Exception as e:
        logger.error("❌ [失败]  %.2fs  %s", time.time() - t0, e, exc_info=True)
        logger.info("━" * 60)
        raise


@router.post("/chat/stream")
async def chat_stream(item: ChatRequest):
    """
    流式问答接口。用 SSE 逐 token 推送给前端。

    注意：create_react_agent() 是同步调用（图构建），
    放在 run_in_executor 里避免短暂阻塞事件循环。
    """
    rid = (item.user_id[:8] if len(item.user_id) >= 8 else item.user_id) + "-" + _uuid.uuid4().hex[:6]
    request_id_var.set(rid)
    t0 = time.time()
    logger.info("[请求开始·流式] user=%s session=%s q=%s",
                item.user_id[:8], item.session_id[:8], len(item.message))

    # 设置当前 user_id，工具内部通过 contextvar 读取
    from app.tools.agent_tools import _current_user_id
    _current_user_id.set(item.user_id)

    # 直接使用全部工具（不再分类路由）
    tools = agent_service.all_tools

    # 先准备好记忆上下文和 agent（同步操作丢线程池）
    memory_context = memory_service.build_memory_context(
        user_id=item.user_id,
        session_id=item.session_id,
        question=item.message,
    )
    from langchain_core.messages import SystemMessage, HumanMessage
    from app.services.react_agent import SYSTEM_PROMPT
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.replace("{memory_context}", memory_context)),
        HumanMessage(content=item.message),
    ]

    loop = asyncio.get_running_loop()
    from langgraph.prebuilt import create_react_agent
    agent = await loop.run_in_executor(
        _executor,
        lambda: create_react_agent(
            model=agent_service.chat_model,
            tools=tools,
        ),
    )

    async def generate():
        full_reply = ""
        tool_calls = []
        last_tool_name = None  # 用于合并连续同名工具
        try:
            async for event in agent.astream_events(
                {"messages": messages},
                version="v2",
            ):
                kind = event.get("event", "")

                # --- Tool call start ---
                if kind == "on_tool_start":
                    tool_data = event.get("data", {}) or {}
                    tool_input = tool_data.get("input", {})
                    name = event.get("name", "unknown_tool")
                    tc = {
                        "name": name,
                        "input": tool_input if isinstance(tool_input, dict) else {"query": str(tool_input)},
                        "output": None,
                        "status": "running",
                    }
                    # 合并连续同名工具：只发一次 TOOL_START
                    if name == last_tool_name:
                        tc["append"] = True
                    else:
                        last_tool_name = name
                        tc["append"] = False
                    tool_calls.append(tc)

                    logger.debug("[流式·工具调用] %s input=%s", name, str(tool_input)[:100])

                    yield f"data: [TOOL_START] {_json.dumps(tc, ensure_ascii=False)}\n\n"

                # --- Tool call end ---
                elif kind == "on_tool_end":
                    output = event.get("data", {}).get("output", "")
                    name = event.get("name", "")
                    # LangGraph's on_tool_end returns a ToolMessage object — extract its .content
                    if hasattr(output, "content"):
                        output_str = output.content
                    else:
                        output_str = str(output) if not isinstance(output, str) else output
                    # Find matching tool call and update
                    is_append = False
                    for tc in reversed(tool_calls):
                        if tc["name"] == name and tc["status"] == "running":
                            tc["output"] = output_str
                            tc["status"] = "done"
                            is_append = tc.get("append", False)

                            logger.debug("[流式·工具完成] %s output_len=%d", name, len(output_str))

                            tool_end_data = {"name": name, "output": output_str, "status": "done", "append": is_append}
                            yield f"data: [TOOL_END] {_json.dumps(tool_end_data, ensure_ascii=False)}\n\n"
                            break

                # --- Text streaming ---
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", None)
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # 归一化字面 \n 为真换行（与非流式端点 routes.py:95 保持一致）
                        text = chunk.content.replace('\\\\n', '\n')
                        full_reply += text
                        # 用 JSON 包装，避免 text 里的真换行/方括号破坏 SSE 协议
                        # （之前裸 yield "data: {text}\n\n"，text 含换行时前端 split('\n')
                        #  会把换行后的内容当非 data 行丢掉）
                        yield f"data: {_json.dumps({'t': text}, ensure_ascii=False)}\n\n"

                # --- Tool error ---
                elif kind == "on_tool_error":
                    name = event.get("name", "")
                    for tc in reversed(tool_calls):
                        if tc["name"] == name and tc["status"] == "running":
                            tc["status"] = "error"
                            logger.debug("[流式·工具错误] %s", name)
                            tool_end_data = {"name": name, "status": "error"}
                            yield f"data: [TOOL_END] {_json.dumps(tool_end_data, ensure_ascii=False)}\n\n"
                            break

            # --- Emit sources ---
            kb_sources = []
            for tc in tool_calls:
                if tc["name"] == "search_knowledge_base" and tc["output"]:
                    found = re.findall(r"来自《(.+?)》", tc["output"])
                    kb_sources.extend(found)
            sources = list(dict.fromkeys(kb_sources))

            if sources:
                yield f"data: [SOURCES] {_json.dumps(sources, ensure_ascii=False)}\n\n"

            # ── 先发右栏上下文，再发 [DONE] ──
            # 顺序很重要：前端在 [DONE] 分支会调 showContextInPanel(msg)，
            # 此时 msg.context 必须已经赋值，否则右栏永远空白
            from app.services.react_agent import SYSTEM_PROMPT
            full_prompt = SYSTEM_PROMPT.replace("{memory_context}", memory_context)
            full_context = f"{full_prompt}\n\n[用户消息] {item.message}"
            yield f"data: [CONTEXT] {_json.dumps(full_context, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

            logger.info("[请求完成·流式] 耗时=%.2fs reply_len=%d tool_calls=%d sources=%d",
                        time.time() - t0, len(full_reply), len(tool_calls), len(sources))

            memory_service.after_response(
                user_id=item.user_id,
                session_id=item.session_id,
                user_msg=item.message,
                bot_reply=full_reply,
            )

        except Exception as e:
            logger.error("[请求失败·流式] 耗时=%.2fs error=%s", time.time() - t0, e, exc_info=True)
            # 即使出错也保存记忆，避免"聊一句忘一句"
            try:
                memory_service.after_response(
                    user_id=item.user_id,
                    session_id=item.session_id,
                    user_msg=item.message,
                    bot_reply=full_reply or "（回复失败）",
                )
            except Exception:
                pass
            yield f"data: {_json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# 文件上传接口
# ============================================================

ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".json", ".csv", ".log", ".py", ".js", ".html", ".css",
    ".pdf", ".docx", ".pptx", ".xlsx",
}


def _parse_file(filepath: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".txt", ".md", ".json", ".csv", ".log", ".py", ".js", ".html", ".css"}:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(filepath)
        return result.text_content
    except Exception:
        return f"[解析失败] 不支持的文件类型：{ext}"


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
):
    # P0-5: user_id 严格校验，防止路径穿越
    import re as _re
    if not _re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", user_id or ""):
        return {"ok": False, "error": "user_id 只允许字母、数字、下划线、短横线"}
    # P0-5: filename 清洗，只取 basename
    _safe_filename = os.path.basename(file.filename or "unknown.txt")
    ext = os.path.splitext(_safe_filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {"ok": False, "error": f"不支持的类型：{ext}"}

    user_dir = os.path.join("uploads", user_id)
    os.makedirs(user_dir, exist_ok=True)
    # P0-5: 完全丢弃客户端 filename，只用 uuid
    safe_name = f"{_uuid.uuid4().hex}{ext}"
    save_path = os.path.join(user_dir, safe_name)
    # P0-5: 最终路径校验
    _uploads_root = os.path.abspath("uploads")
    _final_path = os.path.abspath(save_path)
    if not _final_path.startswith(_uploads_root + os.sep):
        return {"ok": False, "error": "非法路径"}
    # P0-6: 分块写入，20MB 限制
    MAX_FILE_SIZE = 20 * 1024 * 1024
    _total = 0
    with open(save_path, "wb") as f:
        while True:
            _chunk = await file.read(64 * 1024)
            if not _chunk:
                break
            _total += len(_chunk)
            if _total > MAX_FILE_SIZE:
                f.close()
                os.remove(save_path)
                return {"ok": False, "error": "文件超过 20MB 限制"}
            f.write(_chunk)
    content = None

    try:
        text = _parse_file(save_path, file.filename or "unknown.txt")
        if not text or text.startswith("[解析失败"):
            return {"ok": False, "error": text}

        from app.tools.agent_tools import _get_user_store
        import hashlib
        store = _get_user_store(user_id)

        # 去重：按内容 MD5 判断，同名不同内容允许，同内容不同名拒绝
        content_hash = hashlib.md5(text.encode()).hexdigest()
        if store.exists():
            existing = store.search(content_hash, k=1)
            for doc in existing:
                if doc.metadata.get("content_hash") == content_hash:
                    return {"ok": False, "error": f"文件内容与「{doc.metadata.get('title', '未知')}」相同，无需重复上传"}

        store.add_documents([(text[:8000], {"title": file.filename, "source": "user_upload", "user_id": user_id, "content_hash": content_hash})])

        return {"ok": True, "file": file.filename, "size": _total, "chars": len(text)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
