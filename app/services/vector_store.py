import os
import shutil

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from app.core import config
from app.services.document_loader import DocumentLoader
from app.core.logger import get_logger

logger = get_logger("vector_store")


class VectorStore:
    """向量存储：管理 ChromaDB 的创建、加载、检索和写入"""

    def __init__(self, persist_dir=None, data_dir=None):
        """
        persist_dir: ChromaDB 持久化目录（默认 config.CHROMA_GENERAL_DIR）
        data_dir:   .md 文档目录，None 表示不从文件构建（如用户专属库）
        """
        self.persist_dir = persist_dir or config.CHROMA_GENERAL_DIR
        self.data_dir = data_dir
        self.embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    # ========== 构建 / 加载 ==========

    def build_index(self):
        """从 .md 文件读取 → 切分 → 存入 ChromaDB（覆盖旧数据）"""
        if not self.data_dir:
            raise ValueError("data_dir 未设置，无法从文件构建索引")
        logger.info("正在从 %s 构建向量索引...", self.data_dir)
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        loader = DocumentLoader(data_dir=self.data_dir)
        docs = loader.load_and_split()
        self._store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=config.COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"},
        )
        logger.info("向量索引构建完成，共 %d 个文档块", len(docs))
        return self._store

    def load_index(self):
        """从磁盘加载已有 ChromaDB"""
        self._store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=config.COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"},
        )
        return self._store

    def _ensure_store(self):
        """保证 self._store 已初始化（构建或加载）"""
        if not hasattr(self, '_store'):
            if os.path.exists(self.persist_dir) and \
               os.path.isdir(self.persist_dir) and \
               os.listdir(self.persist_dir):
                self._store = self.load_index()
            elif self.data_dir:
                self._store = self.build_index()
            else:
                os.makedirs(self.persist_dir, exist_ok=True)
                self._store = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=config.COLLECTION_NAME,
                    collection_metadata={"hnsw:space": "cosine"},
                )

    # ========== 检索 ==========

    def exists(self):
        """当前 store 是否有数据"""
        if not os.path.exists(self.persist_dir):
            return False
        self._ensure_store()
        try:
            return self._store._collection.count() > 0
        except Exception:
            return False

    def get_retriever(self):
        """获取检索器，k=3"""
        self._ensure_store()
        return self._store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    def search(self, query: str, k: int = 3):
        """直接搜，返回 LangChain Document 列表（cosine 距离越小越相似）"""
        threshold = config.RETRIEVER_THRESHOLD
        self._ensure_store()
        # similarity_search_with_score 返回 [(文档, 距离), ...]
        # cosine 距离：0=完全相似，1=正交，2=完全相反
        # threshold 现在是"cosine 距离上限"，距离 < threshold 才保留
        results = self._store.similarity_search_with_score(query, k=max(k, 20))
        filtered = [doc for doc, score in results if score < threshold]
        return filtered[:k] if filtered else []

    # ========== 写入 ==========

    def add_documents(self, texts_and_metadata):
        """
        向知识库新增文档。
        texts_and_metadata: [(text, metadata_dict), ...]
        """
        self._ensure_store()
        docs = []
        for text, meta in texts_and_metadata:
            docs.append(Document(page_content=text, metadata=meta))
        self._store.add_documents(docs)
