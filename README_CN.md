
<div align="center">
  <h1>AgenticRAG 🧠</h1>
  <p><strong>本地优先的 RAG Agent — 双路检索 + ReAct 推理 + 分级记忆</strong></p>
</div>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/hyx1249207016-netizen/AgenticRAG/actions/workflows/ci.yml">
    <img src="https://github.com/hyx1249207016-netizen/AgenticRAG/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
  <a href="https://python.org">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python" alt="Python">
  </a>
  <a href="https://ollama.com">
    <img src="https://img.shields.io/badge/Ollama-Qwen3.5-5B5?logo=ollama" alt="Ollama">
  </a>
</p>

---

**AgenticRAG** 是一个本地优先、生产级可用的 AI Agent，融合了**双路检索**（语义向量 + BM25）、**ReAct 推理**和**分级记忆**（短期记忆、长期记忆、用户画像）。完全基于 Ollama 本地运行——无需云 API，数据不离开本机，零额外费用。

它不仅是一个校园知识助手，更是一个**通用知识库问答框架**。你可以接入任何知识库——企业文档、个人笔记、科研论文、产品手册——几分钟内得到一个智能 Agent。

---

## 功能特性

| 特性 | 说明 |
|---|---|
| **ReAct Agent** | 基于 LangGraph 的工具调用循环，内置 4 个工具：知识库搜索、网页搜索、网页抓取、时钟 |
| **双路检索** | 语义向量检索（ChromaDB + BGE 嵌入）+ BM25 关键词检索，经 bge-reranker 重排序融合 |
| **分级记忆** | 短期记忆（10 轮对话）+ ChromaDB 长期摘要 + 用户画像——永不丢失上下文 |
| **SSRF 防护** | 双重 IP 校验（Python + nslookup），防止服务端请求伪造攻击 |
| **优雅降级** | 自动兜底：语义搜索 BM25 搜索 直接 LLM 回答，内置熔断和重试 |
| **流式输出** | 基于 SSE 的实时流式响应，通过 FastAPI 提供 |
| **模型无关** | 通过配置文件切换模型OllamaOpenAI 兼容接口或任意 LangChain 支持的 LLM |
| **Docker 支持** | docker-compose 一键部署 |
| **评估工具** | 内置 RAG 准确率评估脚本，支持持续迭代优化 |

---

## 快速开始

### 前置条件

- [Ollama](https://ollama.com) 已安装并运行
- Python 3.10+

### 1. 克隆并安装

```bash
git clone https://github.com/hyx1249207016-netizen/AgenticRAG.git
cd AgenticRAG

python -m venv .venv
.venv\Scripts\activate  # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 下载模型

```bash
ollama pull qwen3.5:0.5b
ollama pull bge-m3:latest
```

创建自定义模型配置：

```bash
ollama create agenticrag-model -f Modelfile.qwen35
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

### 4. 准备知识库

将你的 Markdown 文档放入 `data/` 目录，或直接使用内置的示例数据。

### 5. 启动

```bash
python main.py
```

打开浏览器访问 [http://localhost:8000](http://localhost:8000) 开始对话。

> Docker 用户：运行 `docker compose up -d` 可一键容器化部署。

---

## 配置说明

`.env` 中的关键环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama 服务地址 |
| CHAT_MODEL | agenticrag-model | 对话用的 LLM |
| EMBEDDING_MODEL | bge-m3:latest | 嵌入模型 |
| MEMORY_PERSIST_DIR | ./data/memory_db | 长期记忆存储路径 |

完整配置项见 [.env.example](.env.example)。

---

## 评估

```bash
python eval.py
```

运行 RAG 准确率评估，输出精确率和召回率指标。

---

## 项目结构

```
AgenticRAG/
├── main.py                     # FastAPI 入口
├── app/
│   ├── api/routes.py           # 聊天和文件上传接口
│   ├── core/config.py          # 环境配置
│   ├── services/
│   │   ├── memory_service.py   # 分级记忆系统
│   │   ├── react_agent.py      # LangGraph ReAct Agent
│   │   ├── reranker.py         # bge-reranker 重排序
│   │   └── vector_store.py     # ChromaDB 管理
│   └── tools/agent_tools.py    # 工具实现
├── data/                       # 知识库文档
├── static/                     # Web 前端
├── tests/                      # 测试
└── docker-compose.yml          # Docker 部署
---

## 安全

- **SSRF 防护**：每个网络请求都经过双重 IP 校验Python urllib + 系统 nslookup在建立连接前拦截私有/内网 IP。
- **输入校验**：文件上传做类型检查，API 输入通过 Pydantic schema 校验。
- **无遥测**：零数据收集，完全本地运行。

---

## 参与贡献

欢迎贡献代码！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

---

## 开源协议

[MIT](LICENSE) 2025 AgenticRAG

---

<p align="center">
  <sub>基于 LangGraph、ChromaDB、FastAPI 和 Ollama 构建</sub>
</p>
