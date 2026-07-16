import os

# 项目根目录（Agent 文件夹）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========== Ollama 模型配置 ==========
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")        # 向量模型：把文字转成数字指纹
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen35-campus")  # 对话模型（num_ctx=32k，修复 400 错误）
CHAT_TEMPERATURE = 0.1                # 温度：越低越守规矩（0=死板，1=天马行空）

# ========== ChromaDB 向量库 ==========
CHROMA_GENERAL_DIR = os.path.join(BASE_DIR, "chroma_db", "general")  # 通用知识库（静态 .md）
CHROMA_USER_BASE_DIR = os.path.join(BASE_DIR, "chroma_db")           # 用户专属库父目录（下面 user_{uuid}/）
COLLECTION_NAME = "campus_knowledge"
DATA_DIR = os.path.join(BASE_DIR, "data")  # 知识库 .md 文档目录

# ========== 文档切分参数 ==========
CHUNK_SIZE = 800      # 每段最多xx字
CHUNK_OVERLAP = 150   # 相邻两段重叠xx字（防止切到一半断开）
MIN_CHUNK_LENGTH = 30  # 低于此长度的 chunk 会合并到前一个（避免碎片化）

# ========== 检索参数 ==========
RETRIEVER_K = 30              # 向量检索初筛召回条数（多召回，靠重排筛）
RETRIEVER_THRESHOLD = 0.8     # 【越小越严】cosine 距离上限，范围 0~2（0=完全相似，1=无关，2=相反）
                              #   过滤逻辑：距离 < 此值才保留。
BM25_THRESHOLD = 0.6          # 【越大越严】BM25 归一化分数下限，范围 0~1（1=完美匹配，0=不匹配）
                              #   过滤逻辑：score/max_score >= 此值才保留。
RERANKER_THRESHOLD = -4.0     # 【越大越严】bge-reranker 绝对分数下限，范围 -∞~+∞（>0.3=明显相关）
                              #   过滤逻辑：score >= 此值才保留。
RERANKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "bge-reranker-base")  # 本地模型路径（用 ModelScope 下载）

# ========== 记忆系统配置 ==========
MEMORY_PERSIST_DIR = os.path.join(BASE_DIR, "memory_db")  # 长期记忆的 ChromaDB 目录
MEMORY_COLLECTION = "long_term_memory"                     # 长期记忆的 collection 名
MAX_SHORT_TERM_ROUNDS = 10    # 短期记忆最多保留几轮对话
MAX_LONG_TERM_ENTRIES = 50   # 每个用户的长期记忆最多几条
MEMORY_RETRIEVE_TOP_K = 3    # 记忆检索时返回最相关的几条
MEMORY_SUMMARY_MAX_LEN = 200 # 生成摘要时，回答最多截断多少字
MEMORY_CONTEXT_MAX_LEN = 0  # 短期记忆注入 Prompt 时回答截断长度，0=不截断
MEMORY_BATCH_SIZE = 5       #  攒够多少轮被踢出的对话后，批量生成一次会话级摘要
MEMORY_RECENT_TOP_K = 5     #  时间线检索时，返回最近几条摘要（不管语义）
MEMORY_DEDUP_THRESHOLD = 0.85  # 去重阈值：新摘要与已有相似度>此值则跳过（1=完全相同）
