"""
reranker.py — 轻量重排 + BM25 关键词检索

替代 _grade_docs（不再调 ollama 打分），用 100MB 小模型毫秒级重排。
同时提供 BM25 关键词检索，与向量检索并行。
"""

from typing import List
import os

from langchain_core.documents import Document

from app.core.logger import get_logger
logger = get_logger("reranker")


# ============================================================
# BM25 关键词检索（与向量检索并行）
# ============================================================

class BM25Index:
    """内存级 BM25 关键词索引，每次 rebuild 时从文档列表构建"""

    def __init__(self):
        self._docs: List[Document] = []
        self._bm25 = None

    def rebuild(self, docs: List[Document]):
        """用最新文档列表重建 BM25 索引"""
        from rank_bm25 import BM25Okapi
        import jieba

        self._docs = docs
        # 分词：中文用 jieba，英文按空格
        tokenized = []
        for doc in docs:
            text = doc.page_content
            tokens = list(jieba.cut(text))
            tokenized.append(tokens)
        if not tokenized:
            self._bm25 = None
            logger.warning("[BM25] 文档列表为空，跳过索引构建")
            return
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> List[Document]:
        """BM25 检索，返回 top-k 文档（归一化分数 < BM25_THRESHOLD 的丢掉）"""
        import jieba

        if not self._bm25:
            return []
        tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokens)
        # 按分数降序取 top k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        if not ranked or ranked[0][1] <= 0:
            return []
        # 归一化到 [0,1]：最高分作为分母，低分文档丢掉
        # 不再"score>0 就要"，避免字面词命中导致的垃圾召回
        from app.core import config
        max_score = ranked[0][1]
        return [
            self._docs[i]
            for i, score in ranked[:k]
            if score > 0 and (score / max_score) >= config.BM25_THRESHOLD
        ]


# 全局单例
_bm25_index = BM25Index()


def rebuild_bm25(docs: List[Document]):
    """外部调用：文档库更新后重建 BM25"""
    _bm25_index.rebuild(docs)


def bm25_search(query: str, k: int = 5) -> List[Document]:
    """外部调用：BM25 关键词检索"""
    return _bm25_index.search(query, k)


# ============================================================
# 轻量重排模型
# ============================================================

# 加载失败的哨兵值，区分"未尝试"(None) 和"尝试过但失败"
_RERANKER_LOAD_FAILED = object()


class RerankerService:
    """用 flashrank 的 ms-marco-MiniLM 做重排，100MB，CPU 毫秒级。
    
    设计原则：
    - 优雅降级：加载失败时不抛异常，直接返回原始文档列表的前 top_k 条
    - 哨兵值区分：None=未尝试加载, _LOAD_FAILED=加载失败, 对象=加载成功
    """

    def __init__(self):
        self._reranker = None

    def _ensure_loaded(self):
        """确保 reranker 已加载；若已尝试过（无论成功或失败）则跳过。"""
        if self._reranker is not None:
            return
        try:
            from rerankers import Reranker
            from app.core import config
            # 中文 Reranker：bge-reranker-base（BAAI 中文重排模型，和 bge-m3 配套）
            # 原来用的 ms-marco-MiniLM-L-12-v2 是英文模型，对中文 query-doc 对打分不可信
            self._reranker = Reranker(config.RERANKER_MODEL_PATH, model_type="cross-encoder")
            logger.info("Reranker 模型已加载 (BAAI/bge-reranker-base / cross-encoder)")
        except Exception as e:
            logger.warning(
                "无法加载 Reranker 模型 (BAAI/bge-reranker-base): %s. "
                "重排序功能将降级：直接返回原始检索结果的前 top_k 条。"
                "请安装依赖: pip install 'rerankers[transformers]'",
                e,
            )
            self._reranker = _RERANKER_LOAD_FAILED

    @property
    def is_degraded(self) -> bool:
        """是否处于降级模式（reranker 加载失败，未重排）"""
        self._ensure_loaded()
        return self._reranker is _RERANKER_LOAD_FAILED

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """重排文档列表，返回 top_k。

        若 reranker 模型未加载或加载失败，优雅降级：
        只返回前 min(top_k, 2) 条（减少无关文档），不抛异常。
        """
        if not docs:
            return []
        self._ensure_loaded()

        # 优雅降级：reranker 不可用时，只返回前 2 条（避免硬塞一堆无关文档）
        if self._reranker is _RERANKER_LOAD_FAILED:
            return docs[:min(top_k, 2)]

        # 传字符串列表，rerankers 自动生成 doc_id=0,1,2...
        passages = [doc.page_content[:500] for doc in docs]
        results = self._reranker.rank(query=query, docs=passages)

        # 收集 (idx, score)
        scored = []
        for r in results:
            idx = r.doc_id if hasattr(r, 'doc_id') else r['doc_id']
            score = r.score if hasattr(r, 'score') else r.get('score', 0)
            if 0 <= idx < len(docs):
                scored.append((idx, score))

        if not scored:
            return []

        # 绝对阈值过滤：bge-reranker 的 logit 分数 > 0.3 基本相关
        # 不用 max 归一化（会误杀相关文档），直接用绝对阈值
        # 哪怕最后只剩 0 条，也比硬塞无关文档强（0 条会触发"未找到"）
        from app.core import config
        reranked = [docs[idx] for idx, score in scored if score >= config.RERANKER_THRESHOLD]
        return reranked[:top_k]


# 全局单例
_reranker = RerankerService()


def rerank(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """外部调用：重排"""
    return _reranker.rerank(query, docs, top_k)


def is_degraded() -> bool:
    """外部调用：reranker 是否处于降级模式"""
    return _reranker.is_degraded
