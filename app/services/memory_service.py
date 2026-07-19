import uuid
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from app.core import config
from app.core.logger import get_logger

logger = get_logger("memory")

class MemoryService:
    """记忆服务：短期记忆（上下文窗口）+ 长期记忆（ChromaDB 摘要）+ 用户画像"""

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
        self.short_term: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        self._state_file = os.path.join(
            os.path.dirname(config.MEMORY_PERSIST_DIR), "memory_state.json"
        )
        self.pending_buffer: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        self.user_profiles: Dict[str, dict] = {}
        self.summarize_llm = ChatOllama(
            model=config.CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.CHAT_TEMPERATURE,
        )
        self._load_state()
        self._lock = threading.Lock()
        self._data_lock = threading.RLock()
        self._bg_executor = ThreadPoolExecutor(max_workers=1)

    def get_short_term(self, user_id: str, session_id: str) -> List[Tuple[str, str]]:
        with self._data_lock:
            return self.short_term.get((user_id, session_id), [])

    def add_short_term(self, user_id: str, session_id: str,
                       user_msg: str, bot_reply: str) -> List[Tuple[str, str]]:
        with self._data_lock:
            key = (user_id, session_id)
            if key not in self.short_term:
                self.short_term[key] = []
            self.short_term[key].append((user_msg, bot_reply))
            kicked_out = []
            if len(self.short_term[key]) > config.MAX_SHORT_TERM_ROUNDS:
                excess = len(self.short_term[key]) - config.MAX_SHORT_TERM_ROUNDS
                kicked_out = self.short_term[key][:excess]
                self.short_term[key] = self.short_term[key][-config.MAX_SHORT_TERM_ROUNDS:]
            return kicked_out

    def generate_batch_summary(self, conversations: List[Tuple[str, str]]) -> str:
        if not conversations:
            return ""
        dialogue_text = ""
        for i, (user_msg, bot_reply) in enumerate(conversations, 1):
            dialogue_text += (
                f"第{i}轮：\n"
                f"用户：{user_msg}\n"
                f"助手：{bot_reply[:config.MEMORY_SUMMARY_MAX_LEN]}\n\n"
            )
        prompt = (
            "请用一段话（不超过 120 字）概括以下多轮对话的核心内容。\n"
            "要求：按时间顺序描述用户依次遇到了哪些问题，以及分别如何解决的。\n"
            f"{dialogue_text}"
        )
        response = self.summarize_llm.invoke(prompt)
        return response.content.strip()

    def assess_importance(self, conversations: List[Tuple[str, str]]) -> str:
        if not conversations:
            return "trivial"
        if len(conversations) <= 1:
            msg = conversations[0][0].strip().lower()
            trivial_patterns = ["你好", "hello", "hi", "谢谢", "测试", "今天几号",
                                "在吗", "你是谁", "你能做什么"]
            for p in trivial_patterns:
                if msg == p or msg.startswith(p + "，") or msg.startswith(p + ","):
                    return "trivial"
        dialogue_text = ""
        for i, (user_msg, bot_reply) in enumerate(conversations, 1):
            dialogue_text += f"第{i}轮：用户：{user_msg[:100]}\n助手：{bot_reply[:100]}\n"
        prompt = (
            "判断以下对话的重要性，只返回一个词：important / normal / trivial\n\n"
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
            return "normal"

    def extract_profile_fields(self, summary: str) -> dict:
        if not summary:
            return {}
        prompt = (
            "从以下校园对话摘要中提取用户特征信息。\n"
            "只提取明确提到的信息，没提到的字段留空。"
            "返回严格 JSON 格式（不要任何额外文字）：\n\n"
            '{"os": "用户的操作系统(Windows/macOS/未提及)",\n'
            ' "services": ["使用的服务或软件列表"],\n'
            ' "topics": ["本次涉及的问题主题"],\n'
            ' "other": "其他有用信息(未提及则空)"}\n\n'
            f"摘要：{summary}\n\nJSON："
        )
        try:
            response = self.summarize_llm.invoke(prompt)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)
        except Exception:
            return {}

    def update_user_profile(self, user_id: str, fields: dict):
        with self._data_lock:
            if not fields:
                return
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "os": "", "services": [], "topics": [],
                    "other": "", "last_updated": "",
                }
            profile = self.user_profiles[user_id]
            if fields.get("os") and fields["os"] != "未提及":
                profile["os"] = fields["os"]
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
                    if len(profile["topics"]) > 10:
                        profile["topics"] = profile["topics"][-10:]
            if fields.get("other"):
                profile["other"] = fields["other"]
            profile["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            self._save_state()

    def build_user_profile(self, user_id: str) -> str:
        with self._data_lock:
            profile = self.user_profiles.get(user_id)
            if not profile:
                return ""
            parts = []
            if profile.get("os"):
                parts.append(f"操作系统：{profile['os']}")
            if profile.get("services"):
                parts.append(f"常用服务：{'、'.join(profile['services'])}")
            if profile.get("topics"):
                parts.append(f"近期关注：{'、'.join(profile['topics'])}")
            if profile.get("other"):
                parts.append(f"备注：{profile['other']}")
            if not parts:
                return ""
            return "=== 用户画像 ===\n" + " | ".join(parts)

    def save_long_term(self, user_id: str, session_id: str,
                       summary: str, importance: str = "normal"):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        doc_id = str(uuid.uuid4())
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
        results = self.memory_store.get(
            where={"user_id": user_id},
            include=["metadatas"],
        )
        ids = results.get("ids", [])
        if len(ids) > config.MAX_LONG_TERM_ENTRIES:
            metadatas = results.get("metadatas", [])
            sorted_pairs = sorted(
                zip(metadatas, ids),
                key=lambda x: x[0].get("timestamp", "")
            )
            to_delete = [pair[1] for pair in sorted_pairs[:len(ids) - config.MAX_LONG_TERM_ENTRIES]]
            if to_delete:
                self.memory_store.delete(ids=to_delete)

    def build_memory_context(self, user_id: str, session_id: str,
                              question: str) -> str:
        """为 Prompt 组装记忆上下文"""
        parts = []

        profile_text = self.build_user_profile(user_id)
        if profile_text:
            parts.append(profile_text)

        short = self.get_short_term(user_id, session_id)
        if short:
            history = "\n".join(
                f"用户：{msg}\n助手：{rpl[:config.MEMORY_CONTEXT_MAX_LEN] if config.MEMORY_CONTEXT_MAX_LEN else rpl}"
                for msg, rpl in short[-config.MAX_SHORT_TERM_ROUNDS:]
            )
            parts.append(f"=== 近期对话 ===\n{history}")

        long_results = self.memory_store.similarity_search(
            question, k=config.MEMORY_RETRIEVE_TOP_K,
            filter={"user_id": user_id},
        )
        if long_results:
            recent_results = self.memory_store.get(
                where={"user_id": user_id},
                include=["metadatas", "documents"],
            )
            recent_ids = []
            if recent_results.get("ids"):
                pairs = sorted(
                    zip(recent_results["metadatas"], recent_results["ids"], recent_results["documents"]),
                    key=lambda x: x[0].get("timestamp", ""),
                    reverse=True,
                )
                recent_ids = set(p[1] for p in pairs[:config.MEMORY_RECENT_TOP_K])

            seen_summaries = set()
            deduped = []
            for doc in long_results:
                if doc.page_content not in seen_summaries:
                    seen_summaries.add(doc.page_content)
                    deduped.append(doc)
                elif doc.id in recent_ids:
                    deduped.append(doc)

            seen_ids = set(d.id for d in long_results)
            if recent_ids:
                for meta, doc_id, doc_text in pairs[:config.MEMORY_RECENT_TOP_K]:
                    if doc_id not in seen_ids:
                        deduped.append(Document(page_content=doc_text, metadata=meta))

            if deduped:
                memories = "\n".join(
                    f"[{d.metadata.get('timestamp', '?')}] {d.page_content}"
                    for d in deduped[:config.MEMORY_RETRIEVE_TOP_K + config.MEMORY_RECENT_TOP_K]
                )
                parts.append(f"=== 历史相关记录 ===\n{memories}")

        return "\n\n".join(parts)

    def after_response(self, user_id: str, session_id: str,
                        user_msg: str, bot_reply: str):
        """Agent 回答后：更新短期 + 异步处理长期记忆"""
        kicked = self.add_short_term(user_id, session_id, user_msg, bot_reply)
        if kicked:
            self._bg_executor.submit(self._process_batch_pipeline, user_id, session_id, kicked)

    def _process_batch_pipeline(self, user_id: str, session_id: str,
                                 kicked: List[Tuple[str, str]]):
        """后台线程：重要性评估 → 摘要 → 画像 → 持久化"""
        with self._lock:
            key = (user_id, session_id)
            if key not in self.pending_buffer:
                self.pending_buffer[key] = []
            self.pending_buffer[key].extend(kicked)

            if len(self.pending_buffer[key]) < config.MEMORY_BATCH_SIZE:
                return

            batch = self.pending_buffer.pop(key)

        try:
            importance = self.assess_importance(batch)
            if importance == "trivial":
                return

            summary = self.generate_batch_summary(batch)
            if not summary:
                return

            if self._dedup_check(user_id, summary):
                return

            self.save_long_term(user_id, session_id, summary, importance)
            fields = self.extract_profile_fields(summary)
            if fields:
                self.update_user_profile(user_id, fields)
        except Exception as e:
            logger.error("[记忆流水线] 失败: %s", e, exc_info=True)

    def _dedup_check(self, user_id: str, summary: str) -> bool:
        try:
            existing = self.memory_store.similarity_search(
                summary, k=1,
                filter={"user_id": user_id},
            )
            if existing:
                from sklearn.metrics.pairwise import cosine_similarity
                query_emb = self.embeddings.embed_query(summary)
                existing_emb = self.embeddings.embed_query(existing[0].page_content)
                sim = cosine_similarity([query_emb], [existing_emb])[0][0]
                if sim > config.MEMORY_DEDUP_THRESHOLD:
                    return True
        except Exception:
            pass
        return False

    def _save_state(self):
        try:
            state = {
                "pending_buffer": {
                    f"{k[0]}||{k[1]}": v for k, v in self.pending_buffer.items()
                },
                "user_profiles": self.user_profiles,
            }
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("[记忆持久化] 保存失败: %s", e)

    def _load_state(self):
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            for key_str, val in state.get("pending_buffer", {}).items():
                u, s = key_str.split("||", 1)
                self.pending_buffer[(u, s)] = val
            self.user_profiles.update(state.get("user_profiles", {}))
        except Exception as e:
            logger.warning("[记忆加载] 读取失败: %s", e)
