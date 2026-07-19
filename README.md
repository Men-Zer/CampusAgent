<div align="center">

# CampusAgent 🎓

### 校园 AI 智能助手 — 问选课、查WiFi、找攻略，一个 Agent 搞定

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-✔-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![RAG](https://img.shields.io/badge/RAG-双路召回+重排-green.svg)](#-技术架构)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**English** · [简体中文](README_CN.md)

📸 效果截图（即将添加）

---

</div>

## 📖 简介

**CampusAgent** 是一个面向校园场景的 AI 问答助手。它利用 **RAG（检索增强生成）** + **ReAct Agent** 技术，帮助大学生快速获取校园相关的信息：

- 📚 **选课攻略** — 怎么选选修课？公共课怎么安排？
- 📶 **校园网络** — WiFi 连不上？校园网怎么充值？
- 🎯 **社团活动** — 有哪些社团？怎么加入？
- 🎓 **考研/留学** — 考研时间线？留学怎么准备？
- 📋 **证件办理** — 学生证丢失怎么办？在校证明怎么开？

## ✨ 核心功能

| 功能 | 说明 | 亮点 |
|------|------|------|
| 🔍 **智能问答** | 自然语言提问，Agent 自动决策 | ReAct 推理 + 工具编排 |
| 📖 **RAG 知识库** | 从 50+ 校园攻略文档中检索答案 | 双路召回 + 重排，Recall@5 **89%** |
| 🌐 **联网搜索** | 知识库找不到时自动上网查 | 百度 + Tavily 双引擎降级 |
| 💾 **记忆系统** | 记住对话上下文，聊越多越懂你 | 短期 10 轮 + ChromaDB 长程摘要 |
| 📎 **文件上传** | 上传自己的学习资料进知识库 | 支持 .pdf/.docx/.txt/.md 等 |
| 🎨 **现代 UI** | 暗色/亮色主题，流式输出，思维可视化 | Mermaid 图表 + 思维导图 |

## 🚀 快速开始

### 前置条件

- Python 3.10+
- [Ollama](https://ollama.ai/)（本地 LLM 运行环境）
- 至少 8GB 显存（推荐 16GB）

### 1. 安装 Ollama 模型

```bash
# 1. 安装 Ollama（如果还没装）
# macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Windows: 从 https://ollama.ai/download 下载安装包

# 2. 拉取嵌入模型
ollama pull bge-m3

# 3. 创建对话模型（基于 Qwen3.5）
# 项目根目录已有 Modelfile.qwen35
ollama create qwen35-campus -f Modelfile.qwen35

# 4. 可选：拉取 bge-reranker 重排模型
# 脚本会自动从 ModelScope 下载
```

### 2. 克隆并安装

```bash
git clone https://github.com/Men-Zer/CampusAgent.git
cd CampusAgent

# 创建虚拟环境（推荐）
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置

```bash
# 复制环境变量模板
cp .env.example .env
# Windows: copy .env.example .env

# 按需修改 .env 文件中的配置
# 默认配置即可直接启动（如果 Ollama 在本地默认端口）
```

### 4. 启动

```bash
python main.py
```

打开浏览器访问 [http://localhost:8000](http://localhost:8000)

### 🐳 使用 Docker（推荐）

```bash
docker compose up -d
# 访问 http://localhost:8000
```

> Docker 方式会自动配置所有依赖，无需手动安装 Ollama 和 Python 包。

## 🏗️ 技术架构

```
用户输入
    │
    ▼
┌────────────────────────────────────────────┐
│          ReAct Agent (LangGraph)            │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ 知识库检索│ │ 联网搜索  │ │ 网页抓取/时钟│  │
│  └────┬─────┘ └────┬─────┘ └──────┬─────┘  │
│       │            │               │        │
└───────┼────────────┼───────────────┼────────┘
        ▼            ▼               ▼
┌─────────────────────────────┐
│      双路召回 + 重排         │
│  ┌────────┐  ┌──────────┐   │
│  │ 向量检索 │  │ BM25 关键词│   │  ← Recall@5: 68% → 89%
│  └────┬───┘  └────┬─────┘   │
│       └─────┬─────┘         │
│             ▼               │
│     ┌────────────┐          │
│     │bge-reranker│          │  ← 重排打分
│     └──────┬─────┘          │
└────────────┼────────────────┘
             ▼
┌─────────────────────────────┐
│        LLM 生成回答           │
│     (Qwen3.5 / Ollama)       │
└─────────────────────────────┘
             ▼
  ┌──── 流式输出（SSE）────┐
  │   FastAPI + 前端 UI    │
  └────────────────────────┘
```

### 关键技术指标

| 指标 | 数值 |
|------|------|
| 知识库 Recall@5 | **89%**（纯向量 68% → 双路+重排 89%） |
| 搜索引擎覆盖率 | **99%+**（百度 + Tavily 双引擎降级） |
| API 响应 P50 | **5.2s** |
| 短期记忆窗口 | 10 轮对话 |
| 长程记忆 | ChromaDB 向量持久化 |

## 📁 项目结构

```
CampusAgent/
├── main.py                  # 应用入口
├── requirements.txt         # Python 依赖
├── Modelfile.qwen35         # Ollama 模型配置
├── .env.example             # 环境变量模板
├── Dockerfile               # Docker 构建
├── docker-compose.yml       # Docker 编排
├── app/
│   ├── api/
│   │   ├── routes.py        # API 路由（/chat, /chat/stream, /upload）
│   │   ├── models.py        # 请求/响应模型
│   │   └── error_handler.py # 全局错误处理
│   ├── core/
│   │   ├── config.py        # 配置管理
│   │   └── logger.py        # 日志系统
│   ├── services/
│   │   ├── react_agent.py   # ReAct Agent 核心
│   │   ├── memory_service.py# 记忆系统
│   │   ├── vector_store.py  # ChromaDB 向量库
│   │   ├── reranker.py      # 重排模型
│   │   └── document_loader.py# 文档加载
│   └── tools/
│       └── agent_tools.py   # Agent 工具集
├── data/                    # 校园攻略文档（.md）
├── tests/                   # 测试
├── static/                  # 前端文件
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
└── uploads/                 # 用户上传文件
```

## 🧪 测试与评测

```bash
# 运行单元测试
pytest tests/ -v

# 运行 RAG 评测
python eval.py
```

## 🔒 安全特性

- **SSRF 防护** — 双重 IP 校验（DNS 解析前 + 解析后），拒绝内网访问
- **路径穿越防护** — 用户上传文件路径严格校验
- **文件上传限制** — 类型白名单 + 20MB 上限
- **Prompt 注入缓解** — 结构化指令隔离 + 恶意输入检测

## 🤝 参与贡献

欢迎任何形式的贡献！请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

## ⭐ Star History

如果你觉得这个项目有帮助，欢迎给个 ⭐ 支持！你的 star 是作者持续更新的动力。
