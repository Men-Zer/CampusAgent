"""
memory_service.py — 记忆系统（短期 + 长期 + 画像）

层次结构：
  第一层：短期记忆 — 内存字典，记录最近几轮对话原文
  第二层：长期记忆 — ChromaDB 向量库，存会话级摘要
  第三层：记忆检索 — 语义匹配 + 按时间取最近
  第四层：用户画像 — 从摘要中提取用户特征，逐步积累

数据流：
  用户提问 → Agent 回答 → after_response()
    → 短期记忆（原文）
    → 攒够一批后 → 后台线程生成摘要
    → 重要性分级 → 琐碎丢弃，有价值保留
    → 去重检查 → 重复的跳过
    → 提取画像字段 → 增量更新
    → 存入 ChromaDB 长期记忆
"""

import uuid
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Tuple

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from app.core import config
from app.core.logger import get_logger

logger = get_logger("memory")


class MemoryService:
    """记忆服务：短期 + 长期 + 画像"""

    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

        self.memory_store = Chroma(
            persist_directory=config.MEMORY_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name=config.MEMORY_COLLECTION,
        )

        # 短期记忆（重启丢弃，只管当前会话上下文）
        self.short_term: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

        # 确定持久化文件路径（放在 memory_db 目录旁边，叫 memory_state.json）
        self._state_file = os.path.join(
            os.path.dirname(config.MEMORY_PERSIST_DIR), "memory_state.json"
        )

        # 待打包缓冲区：从短期记忆踢出后暂存，攒够一批就摘要
        # 用 JSON 文件持久化，重启不丢
        self.pending_buffer: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        # 用户画像：从摘要中提取，增量累积
        # 用同一个 JSON 文件持久化，重启不丢
        self.user_profiles: Dict[str, dict] = {}

        # 对话模型
        self.summarize_llm = ChatOllama(
            model=config.CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.CHAT_TEMPERATURE,
        )

        # 启动时加载持久化状态
        self._load_state()

        # 线程锁：保护 pending_buffer 的并发写入
        # after_response（主线程）和 _process_batch_pipeline（后台线程）都会操作它
        self._lock = threading.Lock()  # 保护 pending_buffer
        self._data_lock = threading.RLock()  # 保护 short_term / user_profiles
        self._bg_executor = ThreadPoolExecutor(max_workers=1)  # 串行化记忆流水线

    # ================================================================
    # 短期记忆 — 当前会话的原文对话
    # ================================================================

    def get_short_term(self, user_id: str, session_id: str) -> List[Tuple[str, str]]:
        """
        获取当前会话的短期记忆。
        返回值是一个列表，每项是 (用户消息, 助手回复) 的元组。
        """
        with self._data_lock:
            key = (user_id, session_id)
            # dict.get(key, 默认值)：如果 key 不存在就返回默认值，不会报 KeyError
            return self.short_term.get(key, [])

    def add_short_term(self, user_id: str, session_id: str,
                       user_msg: str, bot_reply: str) -> List[Tuple[str, str]]:
        """
        往短期记忆追加一轮对话。超出上限时踢掉最旧的。

        返回值：被踢出的旧对话列表（不会再被丢弃，而是交给缓冲区等待打包摘要）。
        列表可能为空（还没超出上限时）。
        """
        with self._data_lock:
            key = (user_id, session_id)
            if key not in self.short_term:
                self.short_term[key] = []
            self.short_term[key].append((user_msg, bot_reply))

            kicked_out = []
            if len(self.short_term[key]) > config.MAX_SHORT_TERM_ROUNDS:
                # 超出部分的数量（通常为1，但预留了批量超出的可能）
                excess = len(self.short_term[key]) - config.MAX_SHORT_TERM_ROUNDS
                # 切片取前 excess 条（最旧的）作为踢出内容
                # kicked_out = 列表[:excess]，意为"从开头取 excess 个"
                kicked_out = self.short_term[key][:excess]
                # 保留最近 MAX_SHORT_TERM_ROUNDS 条
                # 列表[-N:] 意为"从倒数第 N 个取到最后"
                self.short_term[key] = self.short_term[key][-config.MAX_SHORT_TERM_ROUNDS:]
            return kicked_out

    # ================================================================
    # 长期记忆 — 批量摘要生成与存储
    # ================================================================

    def generate_batch_summary(self, conversations: List[Tuple[str, str]]) -> str:
        """
        把多轮对话压缩成一段会话级摘要。

        与逐轮摘要的区别：
          逐轮：10轮对话 → 10条独立摘要，彼此不知道对方存在
          批量：10轮对话 → 1段完整的因果叙事，"先问了A，接着问了B..."

        conversations 格式：[(用户消息, 助手回复), ...]
        """
        if not conversations:
            return ""

        # enumerate(conversations, 1) → 从 1 开始编号（不是从0），用于生成"第1轮:"这样的文本
        dialogue_text = ""
        for i, (user_msg, bot_reply) in enumerate(conversations, 1):
            # bot_reply[:N] 截断太长的回复，避免 Prompt 超出 LLM 上下文窗口
            dialogue_text += (
                f"第{i}轮：\n"
                f"用户：{user_msg}\n"
                f"助手：{bot_reply[:config.MEMORY_SUMMARY_MAX_LEN]}\n\n"
            )

        # f-string 里放变量：f"{dialogue_text}" 会把多行对话拼接进 Prompt
        prompt = (
            "请用一段话（不超过 120 字）概括以下多轮对话的核心内容。\n"
            "要求：按时间顺序描述用户依次遇到了哪些问题，以及分别如何解决的。\n"
            "如果有多个问题，用 '随后'、'接着' 等词串联。\n\n"
            f"{dialogue_text}"
        )
        # .invoke(prompt) 是 LangChain 的统一调用接口，底层会发给 Ollama
        # .content 取出 LLM 返回的文本，.strip() 去除首尾空白
        response = self.summarize_llm.invoke(prompt)
        return response.content.strip()

    def assess_importance(self, conversations: List[Tuple[str, str]]) -> str:
        """
        判断一批对话是否值得存入长期记忆。

        返回值：important / normal / trivial

        设计思路：
          - trivial（琐碎）：用户说"你好""测试""今天几号"，存了浪费空间还污染检索
          - normal（普通）：有信息量但非关键，正常存
          - important（重要）：真正解决了问题，或跨多轮追问才解决，值得长期保留

        先用规则快速过滤（避免调 LLM），再用 LLM 精确判断。
        """
        if not conversations:
            return "trivial"

        # 快速路径：单轮短对话用关键词匹配，不走 LLM（省 token 和时间）
        if len(conversations) <= 1:
            # .strip() 去首尾空格，.lower() 转小写统一比较
            msg = conversations[0][0].strip().lower()
            trivial_patterns = ["你好", "hello", "hi", "谢谢", "测试", "今天几号",
                                "在吗", "你是谁", "你能做什么"]
            # in 操作符检查字符串包含：if "你好" in "你好啊" → True
            for p in trivial_patterns:
                if msg == p or msg.startswith(p + "，") or msg.startswith(p + ","):
                    return "trivial"

        # LLM 精确判断：给 LLM 对话摘要 + 判断标准，让它输出一个词
        dialogue_text = ""
        for i, (user_msg, bot_reply) in enumerate(conversations, 1):
            dialogue_text += f"第{i}轮：用户：{user_msg[:100]}\n助手：{bot_reply[:100]}\n"

        prompt = (
            "判断以下对话的重要性，只返回一个词：important / normal / trivial\n\n"
            "判断标准：\n"
            "- important：解决了实际问题，用户反馈有效，或跨多轮追问直到解决\n"
            "- normal：一般性咨询，有信息量但非关键\n"
            "- trivial：纯问候、闲聊、问日期天气、测试性消息\n\n"
            f"{dialogue_text}\n"
            "重要性（只返回一个词）："
        )
        try:
            response = self.summarize_llm.invoke(prompt)
            result = response.content.strip().lower()
            if "important" in result:
                return "important"
            elif "trivial" in result:
                return "trivial"
            else:
                return "normal"
        except Exception:
            return "normal"  # LLM 挂了就按普通处理，宁可多存不漏

    # ================================================================
    # 用户画像 — 从摘要中提取特征并增量累积
    # ================================================================

    def extract_profile_fields(self, summary: str) -> dict:
        """
        从对话摘要中提取用户特征。

        让 LLM 做"信息抽取"：给定一段文字 + JSON 模板，要求它只填提到过的字段。

        返回示例：
        {"os": "Windows", "services": ["校园网", "选课系统"],
         "topics": ["考研", "社团"], "other": "计算机专业"}

        为什么用 LLM 而不是正则：
        "用户使用的是Windows系统"、"系统是Win10"、"他用的win" →
        三种说法都指 Windows，正则根本写不过来，LLM 的语义理解能统一处理。
        """
        if not summary:
            return {}

        prompt = (
            "从以下校园对话摘要中提取用户特征信息。\n"
            "只提取明确提到的信息，没提到的字段留空。"
            "返回严格 JSON 格式（不要任何额外文字）：\n\n"
            '{"os": "用户的操作系统(Windows/macOS/未提及)",\n'
            ' "services": ["使用的服务或软件列表"],\n'
            ' "topics": ["本次涉及的问题主题"],\n'
            ' "other": "其他有用信息(部门、角色、特殊环境等，未提及则空)"}\n\n'
            f"摘要：{summary}\n\n"
            "JSON："
        )
        try:
            response = self.summarize_llm.invoke(prompt)
            raw = response.content.strip()
            # LLM 有时会在 JSON 外面包 ```json ... ``` 的 Markdown 格式
            # 这里做清理：去掉 ``` 包裹
            if raw.startswith("```"):
                # split("```") → ["", "json\n{...}", ""]，取中间那部分
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]  # 去掉 "json" 前缀
                raw = raw.strip()
            # json.loads() 把字符串解析为 Python 字典
            # 如果 LLM 返回的不是合法 JSON，会抛异常，返回空字典
            return json.loads(raw)
        except Exception:
            return {}

    def update_user_profile(self, user_id: str, fields: dict):
        """
        把新提取的特征增量合并到已有画像中。

        合并策略（不同字段用不同方式）：
          - os（操作系统）：新值直接覆盖，因为系统环境基本不变
          - services（使用的服务）：追加去重，用户可能陆续用到多种服务
          - topics（关注主题）：追加去重，保留最近10个
          - other（备注）：新值覆盖旧值
        """
        with self._data_lock:
            if not fields:
                return

            # 新用户：先建一个空白画像模板
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "os": "",
                    "services": [],
                    "topics": [],
                    "other": "",
                    "last_updated": "",
                }

            profile = self.user_profiles[user_id]

            # 只处理非空且非"未提及"的值（LLM 没提到时会返回"未提及"或空）
            if fields.get("os") and fields["os"] != "未提及":
                profile["os"] = fields["os"]

            # list.append() + 去重：只在列表里没有时才追加
            if fields.get("services"):
                svc = fields["services"]
                if isinstance(svc, str):
                    svc = [svc]
                if isinstance(svc, list):
                    for s in svc:
                        if s not in profile["services"]:
                            profile["services"].append(s)

            if fields.get("topics"):
                top = fields["topics"]
                if isinstance(top, str):
                    top = [top]
                if isinstance(top, list):
                    for t in top:
                        if t not in profile["topics"]:
                            profile["topics"].append(t)
                    # 只保留最近10个主题，防止无限增长
                    if len(profile["topics"]) > 10:
                        profile["topics"] = profile["topics"][-10:]

            if fields.get("other"):
                profile["other"] = fields["other"]

            # strftime：把 datetime 对象格式化为可读字符串 "%Y-%m-%d %H:%M" → "2026-07-14 15:30"
            profile["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

            # 持久化，防止重启丢失画像
            self._save_state()

    def build_user_profile(self, user_id: str) -> str:
        """
        把画像字典转成可注入 Prompt 的简短文本。

        输出示例：
        === 用户画像 ===
        操作系统：Windows | 常用服务：选课系统、考研平台 | 近期关注：考研、社团活动
        """
        with self._data_lock:
            profile = self.user_profiles.get(user_id)
            if not profile:
                return ""

            parts = []
            if profile.get("os"):
                parts.append(f"操作系统：{profile['os']}")
            if profile.get("services"):
                # "、".join(列表) → 用中文顿号拼接，如 "选课系统、图书馆服务、WiFi"
                parts.append(f"常用服务：{'、'.join(profile['services'])}")
            if profile.get("topics"):
                parts.append(f"近期关注：{'、'.join(profile['topics'])}")
            if profile.get("other"):
                parts.append(f"备注：{profile['other']}")

            if not parts:
                return ""

            # " | ".join(parts) → 用竖线连接各部分，紧凑不占 token
            return "=== 用户画像 ===\n" + " | ".join(parts)

    # ================================================================
    # ChromaDB 存取操作
    # ================================================================

    def save_long_term(self, user_id: str, session_id: str,
                       summary: str, importance: str = "normal"):
        """
        存入 ChromaDB 长期记忆。

        metadata（元数据）附带信息标签，检索时可以按这些字段过滤：
          - user_id：按用户隔离
          - session_id：排除当前会话时用
          - timestamp：按时间排序
          - importance：重要性等级（留作后续扩展，如只检索 important 的记忆）
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        doc_id = str(uuid.uuid4())  # 生成唯一ID，避免 ChromaDB 写入冲突

        # add_texts：存入文本 + 元数据，ChromaDB 会自动对文本做向量化
        self.memory_store.add_texts(
            texts=[summary],
            ids=[doc_id],
            metadatas=[{
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": now,
                "importance": importance,
            }],
        )
        self._trim_long_term(user_id)

    def _trim_long_term(self, user_id: str):
        """
        超限清理：每个用户最多保留 MAX_LONG_TERM_ENTRIES 条记忆。

        清理策略：按时间排序，删最旧的。

        sorted() 的 key 参数：
          lambda x: x[0].get("timestamp", "") → 对每个 (metadata, id) 元组，
          取 metadata 的 timestamp 字段作为排序依据，没有则用空字符串。
        """
        results = self.memory_store.get(
            where={"user_id": user_id},
            include=["metadatas"],
        )
        ids = results.get("ids", [])
        if len(ids) > config.MAX_LONG_TERM_ENTRIES:
            metadatas = results.get("metadatas", [])
            # zip(metadatas, ids) → 把两个列表"拉链"配对
            # 然后用 sorted() 按 metadata 里的 timestamp 排序
            sorted_pairs = sorted(
                zip(metadatas, ids),
                key=lambda x: x[0].get("timestamp", "")
            )
            to_delete_count = len(sorted_pairs) - config.MAX_LONG_TERM_ENTRIES
            to_delete_ids = [pair[1] for pair in sorted_pairs[:to_delete_count]]
            self.memory_store.delete(ids=to_delete_ids)

    # ================================================================
    # 记忆检索 — 语义路 + 时间路，双路互补
    # ================================================================

    def retrieve_memory(self, user_id: str, session_id: str,
                        question: str, top_k: int = None) -> List[dict]:
        """
        语义检索：用用户当前问题的向量，去历史摘要里找语义最相似的。

        原理：问题 "选课系统登录不了" 和摘要 "之前选课系统密码重置" 语义接近，
        向量距离小，会被检索出来。

        排除当前 session：你在同一个会话里问过的问题不需要"回忆"，
        短期记忆已经覆盖了。
        """
        all_user_mems = self.memory_store.get(
            where={"user_id": user_id},
            include=["metadatas"],
        )
        if not all_user_mems.get("ids"):
            return []

        if top_k is None:
            top_k = config.MEMORY_RETRIEVE_TOP_K

        # 多搜一些（top_k * 2），因为后面还要过滤掉当前 session 的
        results = self.memory_store.similarity_search(
            query=question,
            k=min(top_k * 2, len(all_user_mems["ids"])),
            filter={"user_id": user_id},
        )

        filtered = []
        for doc in results:
            meta = doc.metadata
            # 排除自己的当前会话（避免"回忆起刚才自己说的话"）
            if meta.get("session_id") != session_id:
                filtered.append({
                    "summary": doc.page_content,
                    "timestamp": meta.get("timestamp", "未知时间"),
                    "session_id": meta.get("session_id", "未知会话"),
                })
        return filtered[:top_k]

    def get_recent_memory(self, user_id: str, session_id: str,
                          top_k: int = None) -> List[dict]:
        """
        时间线检索：不管语义，直接取该用户最近的摘要。

        与语义检索的互补关系：
          语义路 = "跟当前问题像不像"
          时间路 = "这个人最近发生了什么"

        场景：用户问"帮我看看最近电脑怎么这么多问题"，
        语义检索不知道该搜什么，时间线检索直接给出最近的活动轨迹。
        """
        if top_k is None:
            top_k = config.MEMORY_RECENT_TOP_K

        all_user_mems = self.memory_store.get(
            where={"user_id": user_id},
            include=["metadatas", "documents"],
        )
        ids = all_user_mems.get("ids", [])
        if not ids:
            return []

        metadatas = all_user_mems.get("metadatas", [])
        documents = all_user_mems.get("documents", [])

        records = []
        # zip(ids, metadatas, documents) → 三个列表按位置配对
        # 如：[(id1, meta1, doc1), (id2, meta2, doc2), ...]
        for doc_id, meta, doc in zip(ids, metadatas, documents):
            if meta.get("session_id") == session_id:
                continue  # 排除当前会话
            records.append({
                "summary": doc,
                "timestamp": meta.get("timestamp", "未知时间"),
                "session_id": meta.get("session_id", "未知会话"),
            })

        # reverse=True → 倒序（最新的在前）
        records.sort(key=lambda x: x["timestamp"], reverse=True)
        return records[:top_k]

    # ================================================================
    # 上下文组装 — 把四段信息拼成给 LLM 看的统一文本
    # ================================================================

    def build_memory_context(self, user_id: str, session_id: str,
                             question: str) -> str:
        """
        组装完整记忆上下文，按认知优先级排列：

        0. 用户画像 — 让 Agent 先知道"这人是谁"
        1. 历史相关记录 — 跟当前问题语义相似的过往摘要
        2. 最近动态 — 这个人最近的整体活动轨迹
        3. 当前对话 — 本次会话的原文上下文

        返回值直接插入 Agent 的 System Prompt 中。
        """
        context_parts = []

        # 0. 用户画像 — 优先级最高，Agent 应该先了解用户背景
        profile = self.build_user_profile(user_id)
        if profile:
            context_parts.append(profile)
            context_parts.append("")  # 空行分隔

        # 1. 语义匹配的历史
        long_term = self.retrieve_memory(user_id, session_id, question)
        if long_term:
            context_parts.append("=== 历史相关记录（语义匹配） ===")
            for i, mem in enumerate(long_term, 1):
                context_parts.append(
                    f"{i}. [{mem['timestamp']}] {mem['summary']}"
                )
            context_parts.append("")

        # 2. 最近动态（按时间）
        recent = self.get_recent_memory(user_id, session_id)
        if recent:
            context_parts.append("=== 最近动态（按时间） ===")
            for i, mem in enumerate(recent, 1):
                context_parts.append(
                    f"{i}. [{mem['timestamp']}] {mem['summary']}"
                )
            context_parts.append("")

        # 3. 当前对话原文
        short_term = self.get_short_term(user_id, session_id)
        if short_term:
            context_parts.append("=== 当前对话 ===")
            for user_msg, bot_reply in short_term:
                context_parts.append(f"用户：{user_msg}")
                # 0 表示不截断，大于 0 则截断到指定长度
                reply = bot_reply if config.MEMORY_CONTEXT_MAX_LEN <= 0 else bot_reply[:config.MEMORY_CONTEXT_MAX_LEN]
                context_parts.append(f"助手：{reply}")
            context_parts.append("")

        if not context_parts:
            return ""

        # "\n".join(列表) → 用换行符把所有段落连成一段完整文本
        return "\n".join(context_parts)

    # ================================================================
    # 去重 — 防止连问相似问题时存重复摘要
    # ================================================================

    def _is_duplicate(self, user_id: str, summary: str) -> bool:
        """
        检查新摘要是否与已有摘要高度重复。

        典型场景：用户连问3次"选课还是报错"，每次生成的摘要几乎一样。
        如果不拦截，会存3条语义几乎相同的记录，污染检索结果。

        实现方式：
          用 ChromaDB 的 similarity_search_with_score 找到跟新摘要
          最相似的那条已有记录，如果相似度超过阈值就拦截。

          注意：similarity_search_with_score 返回的是 distance（距离），
          不是 similarity（相似度）。距离越小 = 越相似。
          转换公式：similarity = 1 / (1 + distance)，映射到 0~1 区间。
        """
        all_user_mems = self.memory_store.get(
            where={"user_id": user_id},
            include=["metadatas"],
        )
        if not all_user_mems.get("ids"):
            return False  # 还没存过任何记忆，肯定不重复

        try:
            # k=1 只搜最相似的那一条即可
            results = self.memory_store.similarity_search_with_score(
                query=summary,
                k=1,
                filter={"user_id": user_id},
            )
            if not results:
                return False

            # results[0] → (Document对象, distance数值)
            doc, distance = results[0]
            similarity = 1 / (1 + distance) if (1 + distance) != 0 else 0.0  # 距离转相似度
            is_dup = similarity >= config.MEMORY_DEDUP_THRESHOLD

            if is_dup:
                # {similarity:.2%} → 格式化为百分比，如 "87.50%"
                logger.info(
                    "[记忆] 去重拦截: 相似度 %.2f%% > 阈值 %.0f%%",
                    similarity * 100, config.MEMORY_DEDUP_THRESHOLD * 100,
                )
            return is_dup
        except Exception as e:
            logger.error("[记忆] 去重检查失败: %s", e)
            return False  # 检查出问题就当不重复，宁可多存不漏

    # ================================================================
    # 后台流水线 — 异步执行摘要→分级→去重→画像→存储
    # ================================================================

    def _process_batch_pipeline(self, user_id: str, session_id: str,
                                 batch: List[Tuple[str, str]], key: tuple):
        """
        在后台线程中执行完整的记忆处理流水线。

        为什么要异步：
          LLM 调一次要 1~3 秒，流程里有摘要、分级、画像三次调用。
          如果同步执行，after_response() 会阻塞 API 响应 5~10 秒，
          用户那边就是"发完消息等半天才响应用户"。

        方案选择：
          用 threading.Thread 而不是 asyncio。
          原因：LangChain 的 Ollama 客户端底层是同步 HTTP 请求，
          asyncio 包不住同步阻塞调用。threading 简单可靠，无需额外依赖。

        daemon=True 的含义：
          守护线程，主进程退出时自动结束，不会被卡住导致程序无法关闭。
        """
        try:
            # ① 批量生成摘要
            summary = self.generate_batch_summary(batch)
            logger.debug("[记忆·后台] 摘要: %s", summary[:80])

            # ② 重要性分级
            importance = self.assess_importance(batch)

            if importance == "trivial":
                logger.info("[记忆·后台] 分级: trivial → 跳过存储")
            else:
                # ③ 去重检查
                if self._is_duplicate(user_id, summary):
                    logger.info("[记忆·后台] 分级: %s → 去重跳过", importance)
                else:
                    logger.info("[记忆·后台] 分级: %s → 通过", importance)

                    # ④ 提取画像字段并更新
                    profile_fields = self.extract_profile_fields(summary)
                    if profile_fields:
                        self.update_user_profile(user_id, profile_fields)

                    # ⑤ 存入长期记忆
                    self.save_long_term(user_id, session_id, summary, importance)

            # 处理完毕，在锁内清空缓冲区并落盘
            with self._lock:
                # 只删除已处理的 batch，保留处理期间新追加的对话
                if key in self.pending_buffer:
                    self.pending_buffer[key] = self.pending_buffer[key][len(batch):]
                self._save_state()

        except Exception as e:
            logger.error("[记忆·后台] 流水线失败: %s", e, exc_info=True)
            # 保留缓冲区数据，下次 after_response 会重新触发流水线
            # 不要 del，否则未摘要的对话会永久丢失
            with self._lock:
                self._save_state()

    # ================================================================
    # 持久化 — pending_buffer 和 user_profiles 落盘
    # ================================================================

    def _load_state(self):
        """
        启动时从 JSON 文件恢复 pending_buffer 和 user_profiles。

        JSON 的 key 必须是字符串，而 pending_buffer 的 key 是 (user_id, session_id)
        元组。存的时候把元组转成字符串 key（如 "user_001||session_A"），
        加载时再还原为元组。
        """
        if not os.path.exists(self._state_file):
            logger.info("[记忆] 未找到持久化文件，从零开始")
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 恢复 pending_buffer：字符串 key → 元组 key
            raw_buffer = data.get("pending_buffer", {})
            for key_str, conversations in raw_buffer.items():
                user_id, session_id = key_str.split("||", 1)
                # conversations 是列表的列表，内层列表转成元组
                self.pending_buffer[(user_id, session_id)] = [
                    tuple(conv) for conv in conversations
                ]

            # 恢复 user_profiles：直接赋值
            self.user_profiles = data.get("user_profiles", {})

            buffer_count = sum(len(v) for v in self.pending_buffer.values())
            logger.info(
                "[记忆] 持久化状态已加载: %d 个缓冲区 (%d 轮待摘要), %d 个用户画像",
                len(self.pending_buffer), buffer_count, len(self.user_profiles),
            )
        except Exception as e:
            logger.error("[记忆] 加载持久化文件失败，数据可能损坏: %s", e)
            # 备份损坏文件，下次保存时自动重建
            try:
                import shutil
                backup = self._state_file + ".broken." + datetime.now().strftime("%Y%m%d%H%M%S")
                shutil.move(self._state_file, backup)
                logger.debug("[记忆] 损坏文件已备份至: %s", os.path.basename(backup))
            except Exception:
                pass

    def _save_state(self):
        """
        把 pending_buffer 和 user_profiles 写入 JSON 文件。

        pending_buffer 的元组 key 转成 "user_id||session_id" 格式的字符串，
        因为 JSON 不支持元组做 key。对话内容也从元组转成列表。
        """
        try:
            raw_buffer = {}
            for (user_id, session_id), conversations in self.pending_buffer.items():
                key_str = f"{user_id}||{session_id}"
                raw_buffer[key_str] = [list(conv) for conv in conversations]

            data = {
                "pending_buffer": raw_buffer,
                "user_profiles": self.user_profiles,
            }

            tmp = self._state_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._state_file)
        except Exception as e:
            logger.error("[记忆] 保存持久化文件失败: %s", e)

    # ================================================================
    # 入口 — API 每次响应后调用
    # ================================================================

    def after_response(self, user_id: str, session_id: str,
                       user_msg: str, bot_reply: str):
        """
        Agent 回答完后调用此方法。

        调用链路（routes.py 里 /chat 和 /chat/stream 的两个端点都会在
        Agent 返回答案后调用这里）。

        执行逻辑：
          1. 存入短期记忆（纯内存操作，毫秒级）
          2. 如果踢出了旧对话，放入待打包缓冲区
          3. 缓冲区攒够数量后，启动后台线程执行流水线
          4. 返回（不阻塞 API 响应）

        这样设计的好处：用户发消息 → Agent 回答 → 立即返回给用户，
        摘要/分级/画像这些耗时操作在后台偷偷跑，用户完全无感。
        """
        try:
            # ① 存入短期记忆，拿到被踢出的旧对话
            kicked_out = self.add_short_term(user_id, session_id, user_msg, bot_reply)

            # ② 有踢出的内容就放入缓冲区
            if kicked_out:
                key = (user_id, session_id)

                with self._lock:
                    if key not in self.pending_buffer:
                        self.pending_buffer[key] = []

                    before_count = len(self.pending_buffer[key])
                    self.pending_buffer[key].extend(kicked_out)
                    after_count = len(self.pending_buffer[key])

                    # 写到磁盘，防止重启丢失
                    self._save_state()

                logger.info("[记忆] 缓冲区: %d → %d 轮待摘要", before_count, after_count)

                # ③ 缓冲区攒够了 → 后台线程跑流水线（放锁外面，不阻塞其他请求）
                if after_count >= config.MEMORY_BATCH_SIZE:
                    batch = list(self.pending_buffer[key])

                    self._bg_executor.submit(
                        self._process_batch_pipeline,
                        user_id, session_id, batch, key,
                    )
                    logger.info(
                        "[记忆] 后台流水线已启动 (线程: %s, %d 轮对话)",
                        threading.current_thread().name, len(batch),
                    )
        except Exception as e:
            logger.error("[记忆] after_response 异常: %s", e, exc_info=True)
