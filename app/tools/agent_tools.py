"""
agent_tools.py — Agent 工具集

每个函数就是一个"工具"，Agent 根据函数的 docstring 描述
自动判断什么时候该调用哪个工具。

添加新工具只需三步：
  1. 写一个函数（带清晰的 docstring）
  2. 在 get_all_tools() 里注册
  3. 在 react_agent.py 的 SYSTEM_PROMPT 里提到它
"""

import socket
import re
import os
import contextvars

from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core import config
from app.core.logger import get_logger

logger = get_logger("agent_tools")


def _clean_text(s):
    """通用清理函数：移除不可见字符，去空格"""
    if not s: return ""
    s = re.sub(r'[\u200c\u200d\u200b\u200e\u200f\ufeff]', '', str(s))
    return s.strip()

# 轻量 LLM，专用于查询改写和文档打分（温度 0，只做判断，不聊天）
_judge_llm = ChatOllama(
    model=config.CHAT_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=0,
)


# ============================================================
# 当前请求的 user_id，由 react_agent.run() 在每次调用前设置
# contextvars 保证线程安全，不同请求互不干扰
_current_user_id: contextvars.ContextVar = contextvars.ContextVar(
    'current_user_id', default=None
)

# ============================================================
# 工具 0A：查询改写（内部使用，不暴露给 Agent）
# ============================================================

def _rewrite_query(query: str) -> str:
    """查询改写 - 已关闭
    诊断发现改写经常把原查询中能命中的关键词改丢（如"选课"→"教务管理"导致选课文档丢失）。
    原查询直接检索效果更好，故关闭改写。如需恢复，把下方 return 注释掉、恢复原逻辑即可。
    """
    return query
    # --- 以下原改写逻辑已停用 ---
    # vague_words = ('那个', '这个', '上次', '之前', '刚才', '它', '他', '她', '还是', '不行')
    # if len(query) > 30 and not any(w in query for w in vague_words):
    #     return query  # 已经够具体了，跳过

    try:
        prompt = (
            "把这句话改写成一个适合搜索引擎的查询词。"
            "补全模糊指代（如把'那个问题'换成具体话题）。"
            "只返回改写后的词，不要解释。\n"
            f"原句：{query}"
        )
        response = _judge_llm.invoke(prompt)
        rewritten = (response.content or "").strip()
        if rewritten and len(rewritten) >= 2 and rewritten != query:
            return rewritten
    except Exception:
        pass
    return query


# ============================================================
# 工具 0B：文档相关性评分（内部使用，不暴露给 Agent）
# ============================================================


_USER_STORES = {}  # {user_id: VectorStore} 缓存，避免重复建连接

_GENERAL_STORE = None  # 通用知识库，只建一次
_BM25_BUILT = False     # BM25 是否已构建


def _get_general_store():
    global _GENERAL_STORE, _BM25_BUILT
    if _GENERAL_STORE is None:
        from app.services.vector_store import VectorStore
        from app.core import config
        _GENERAL_STORE = VectorStore(
            persist_dir=config.CHROMA_GENERAL_DIR,
            data_dir=config.DATA_DIR,
        )
        if not _BM25_BUILT:
            _build_bm25_for_general_store()
            _BM25_BUILT = True
    return _GENERAL_STORE


def _build_bm25_for_general_store():
    """从通用知识库的原始文档构建 BM25 索引（启动时执行一次）"""
    try:
        from app.services.reranker import rebuild_bm25
        from langchain_community.document_loaders import TextLoader
        from app.core import config
        import glob

        all_docs = []
        for pattern in ["*.md", "*.txt"]:
            for filepath in glob.glob(os.path.join(config.DATA_DIR, pattern)):
                try:
                    loader = TextLoader(filepath, encoding="utf-8")
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception:
                    pass

        if all_docs:
            rebuild_bm25(all_docs)
    except Exception as e:
        logger.error("[BM25] 构建失败", exc_info=True)


def _get_user_store(user_id):
    if user_id not in _USER_STORES:
        from app.services.vector_store import VectorStore
        from app.core import config
        user_dir = os.path.join(config.CHROMA_USER_BASE_DIR, f"user_{user_id}")
        _USER_STORES[user_id] = VectorStore(persist_dir=user_dir)
    return _USER_STORES[user_id]


def search_knowledge_base(question: str) -> str:
    """
    搜索校园知识库，返回与问题相关的文档片段。
    当用户问选课、校园WiFi、社团、考试、考研、实习等问题时使用此工具。
    会自动同时搜索通用校园攻略和属于当前用户的学习资料。
    注意：此工具只返回检索到的资料，不会直接回答问题。
    """
    from app.services.vector_store import VectorStore
    from app.core import config

    result_parts = []
    step_log = []  # 工作流可视化：记录每一步做了什么

    # ── 步骤 1：查询改写 ──
    rewritten = _rewrite_query(question)
    if rewritten != question:
        step_log.append(f"[STEP:查询改写] \"{question}\" → \"{rewritten}\"")
    else:
        step_log.append(f"[STEP:查询改写] 查询已足够具体，无需改写")
    search_query = rewritten

    # ── 步骤 2：向量检索 + 关键词检索（并行双路）──
    step_log.append(f"[STEP:双路检索] 向量 {config.RETRIEVER_K} 条 + 关键词 5 条...")

    all_docs = []  # 收集全部候选，统一重排

    # --- 向量检索（通用库） ---
    try:
        general = _get_general_store()
        gen_docs = general.search(search_query, k=config.RETRIEVER_K)
        if gen_docs:
            for doc in gen_docs:
                doc.metadata["source_type"] = "通用"
                doc.metadata["search_method"] = "向量"
                all_docs.append(doc)
    except Exception as e:
        result_parts.append(f"(通用库检索异常: {e})")

    # --- 关键词检索（通用库 BM25） ---
    try:
        from app.services.reranker import bm25_search
        kw_docs = bm25_search(search_query, k=10)
        # 去重：和向量检索结果比对
        seen_contents = {(d.page_content or "")[:100] for d in all_docs}
        added = 0
        for doc in kw_docs:
            key = doc.page_content[:100]
            if key not in seen_contents:
                seen_contents.add(key)
                doc.metadata["source_type"] = "通用"
                doc.metadata["search_method"] = "关键词"
                all_docs.append(doc)
                added += 1
        step_log.append(f"[STEP:双路检索] 向量 + 关键词共召回 {len(all_docs)} 条（关键词补充 {added} 条）")
    except Exception as e:
        step_log.append(f"[STEP:双路检索] 关键词检索跳过: {e}")

    # --- 用户专属知识库（仅向量） ---
    user_id = _current_user_id.get()
    if user_id:
        try:
            user_store = _get_user_store(user_id)
            if user_store.exists():
                user_docs = user_store.search(search_query, k=config.RETRIEVER_K)
                seen = set()
                for doc in user_docs:
                    key = (doc.metadata.get("title", ""), (doc.page_content or "")[:80])
                    if key not in seen:
                        seen.add(key)
                        doc.metadata["source_type"] = "个人"
                        doc.metadata["search_method"] = "向量"
                        all_docs.append(doc)
        except Exception as e:
            result_parts.append(f"(个人库检索异常: {e})")

    # ── 步骤 3：轻量重排模型打分（替代 _grade_docs 的 ollama 调用）──
    if len(all_docs) > 1:
        step_log.append("[STEP:重排模型] 小模型正在重新打分（毫秒级）...")
        try:
            from app.services.reranker import rerank, is_degraded
            all_docs = rerank(question, all_docs, top_k=4)
            if is_degraded():
                # 降级模式：reranker 不可用，结果未经重排，可信度低
                step_log.append(f"[STEP:重排模型] [降级模式：未重排] reranker 不可用，保留前 {len(all_docs)} 条原始结果")
            else:
                step_log.append(f"[STEP:重排模型] 保留 Top {len(all_docs)} 条")
        except Exception as e:
            step_log.append(f"[STEP:重排模型] 重排失败({e})，降级使用原始排序，保留 Top 4")
            all_docs = all_docs[:4]
    else:
        step_log.append("[STEP:重排模型] 仅 1 条候选，跳过重排")

    # ── 步骤 4：格式化输出 ──
    # 输出格式标准化：每条文档块末尾追加 "来自《标题》"
    # 这样 routes.extract_sources() 和前端 extractSourcesFromContent() 的
    # r"来自《(.+?)》" 正则才能匹配到，否则 sources 永远是空数组
    if not all_docs:
        result_parts.append("知识库中没有找到相关资料。")
    else:
        for doc in all_docs[:4]:  # 最多返回 4 条（少而精）
            stype = doc.metadata.get("source_type", "通用")
            raw_source = doc.metadata.get("source") or "未知来源"
            title = doc.metadata.get("title", "")
            if title:
                label = f"[{stype}] {title}"
                clean_title = title
            else:
                stem = os.path.splitext(os.path.basename(raw_source))[0]
                clean_title = stem.replace('campus_', '').replace('_', ' ')
                label = f"[{stype}] {clean_title}"
            # 末尾追加来源标记，供 sources 提取正则匹配
            result_parts.append(f"--- {label} ---\n{doc.page_content}\n来自《{clean_title}》")

    # 步骤日志放在最前面（前端可以解析 [STEP:xxx] 标记做可视化）
    return "\n".join(step_log) + "\n\n" + "\n\n".join(result_parts)


# ============================================================
# 工具 2：联网搜索
# ============================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _baidu_search(query: str, max_results: int = 5) -> str:
    """百度搜索（baidusearch 库）。被限流或无结果时返回空串，由上层降级到 Tavily。"""
    try:
        from baidusearch.baidusearch import search
        import re, urllib.parse
        results = search(query, num_results=max_results + 3)
        if not results:
            return ""

        # 过滤：跳过百度 AI 总结/大家还在搜/纯相关词列表/相对路径 URL
        skip_keywords = ["AI总结", "大家还在搜", "百度快照", "相关搜索", "为你推荐"]
        valid = []
        seen = set()
        for r in results:
            title = r.get("title", "") or ""
            url = r.get("url", "") or ""
            if not title or not url: continue
            if url.startswith("/s?") or url.startswith("#"): continue
            if any(kw in title for kw in skip_keywords): continue
            if title in seen: continue
            seen.add(title)
            try: url = urllib.parse.unquote(url)
            except: pass
            valid.append(r)
            if len(valid) >= max_results: break

        if not valid:
            return ""

        text = ""
        for i, r in enumerate(valid, 1):
            title = _clean_text(r.get("title", ""))
            abstract = _clean_text(r.get("abstract", ""))
            url = _clean_text(r.get("url", ""))
            text += f"[{i}] {title}\n{abstract}\n来源: {url}\n\n"
        return text
    except Exception as e:
        logger.warning("[web_search·百度] 异常: %s", e)
        raise


def _tavily_search(query: str, max_results: int = 5) -> str:
    """Tavily 搜索降级方案（AI 专用，每月 1000 次免费）。
    百度被限流时兜底使用，返回结构化结果。"""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key="")
        response = client.search(query, max_results=max_results + 3)

        if not response.get("results"):
            return ""

        text = ""
        for i, result in enumerate(response["results"][:max_results], 1):
            title = _clean_text(result.get("title", ""))
            content = _clean_text(result.get("content", ""))
            url = _clean_text(result.get("url", ""))
            if not title or not url:
                continue
            text += f"[{i}] {title}\n{content}\n来源: {url}\n\n"
        return text
    except Exception as e:
        logger.warning("[web_search·Tavily] 异常: %s", e)
        return ""


def web_search(query: str) -> str:
    """
    在互联网上搜索信息（天气、新闻、考研政策、竞赛信息等）。
    当知识库没有相关信息、或用户问校园以外的通用问题时使用此工具。

    搜索策略：优先用百度（baidusearch），百度被限流或无结果时降级到 Tavily。
    百度免费但可能被限流，Tavily 每月 1000 次免费（AI 专用，质量好）。
    """
    # 第一优先：百度（中文质量最好）
    try:
        result = _baidu_search(query, max_results=5)
        if result:
            logger.debug("[web_search] 使用百度，query=%s", query[:40])
            return result
    except Exception:
        pass

    # 降级：Tavily（百度被限流或无结果时）
    logger.info("[web_search] 百度无结果，降级到 Tavily，query=%s", query[:40])
    result = _tavily_search(query, max_results=5)
    if result:
        logger.debug("[web_search] 使用 Tavily，query=%s", query[:40])
        return result

    return "未搜索到相关结果。"


def get_current_time() -> str:
    """返回当前日期和时间。当用户问"今天几号"、"现在几点"时使用。"""
    from datetime import datetime
    now = datetime.now()
    return f"{now.year}年{now.month}月{now.day}日 {now.strftime('%H:%M:%S')} 星期{['一','二','三','四','五','六','日'][now.weekday()]}"


def fetch_webpage(url: str) -> str:
    """获取指定网页的正文内容（前3000字）。当web_search搜到链接后需要阅读全文时使用。"""
    import urllib.request
    import re
    # P1-16: SSRF 防护 - 只允许 http/https，拒绝私网/环回
    from urllib.parse import urlparse
    import ipaddress
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "仅支持 http/https 协议"
    hostname = parsed.hostname or ""
    # SSRF 防护：拒绝内网/环回/链路本地地址
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified:
            return "禁止访问内网/本地地址"
    except ValueError:
        # 域名情况：检查可疑后缀
        blocked_hosts = ("localhost", "metadata.google.internal", "metadata",
                         "metadata.aws.internal", "169.254.169.254")
        blocked_suffixes = (".internal", ".local", ".localhost", ".localdomain")
        hostname_lower = hostname.lower()
        if hostname_lower in blocked_hosts or hostname_lower.endswith(blocked_suffixes):
            return "禁止访问内网地址"
        # DNS 解析后再检查 IP（防止 DNS 重绑定攻击）
        try:
            resolved_ips = socket.getaddrinfo(hostname, None)
            for family, _, _, _, sockaddr in resolved_ips:
                resolved_ip = ipaddress.ip_address(sockaddr[0])
                if resolved_ip.is_private or resolved_ip.is_loopback or resolved_ip.is_link_local or resolved_ip.is_unspecified:
                    return "禁止访问内网/本地地址（DNS 解析后）"
        except Exception:
            pass  # DNS 解析失败不阻塞，后续 urlopen 会自然报错
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read(8 * 1024 * 1024).decode("utf-8", errors="ignore")
        # 去掉 script/style 标签
        html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.S|re.I)
        html = re.sub(r'<[^>]+>', ' ', html)          # 去标签
        html = re.sub(r'\s+', ' ', html).strip()       # 合并空白
        return html[:3000] if len(html) > 3000 else html
    except Exception as e:
        return f"获取网页失败: {e}"


def get_all_tools():
    """返回所有工具的列表，供 Agent 使用。"""
    return [
        StructuredTool.from_function(search_knowledge_base),
        StructuredTool.from_function(web_search),
        StructuredTool.from_function(get_current_time),
        StructuredTool.from_function(fetch_webpage),
    ]
