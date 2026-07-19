<div align="center">

# CampusAgent 🎓

### 校园 AI 智能助手 — 问选课、查WiFi、找攻略，一个 Agent 搞定

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-%E2%9C%94-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![RAG](https://img.shields.io/badge/RAG-%E5%8F%8C%E8%B7%AF%E5%8F%AC%E5%9B%9E%2B%E9%87%8D%E6%8E%92-green.svg)](#-%E6%8A%80%E6%9C%AF%E6%9E%B6%E6%9E%84)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

</div>

## 📖 简介

CampusAgent 是一个面向校园场景的 AI 问答助手。它利用 RAG（检索增强生成）+ ReAct Agent 技术，帮助大学生快速获取校园相关信息。

- 📚 **选课攻略** — 怎么选选修课？公共课怎么安排？
- 📶 **校园网络** — WiFi 连不上？校园网怎么充值？
- 🎯 **社团活动** — 有哪些社团？怎么加入？
- 🎓 **考研/留学** — 考研时间线？留学怎么准备？
- 📋 **证件办理** — 学生证丢失怎么办？在校证明怎么开？

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| 🔍 智能问答 | 自然语言提问，Agent 自动决策 |
| 📖 RAG 知识库 | 50+ 校园攻略文档，双路召回 + 重排，Recall@5 89% |
| 🌐 联网搜索 | 知识库不够时自动上网查，百度 + Tavily 双引擎 |
| 💾 记忆系统 | 记住上下文，聊越多越懂你 |
| 📎 文件上传 | 上传学习资料进知识库 |
| 🎨 现代 UI | 暗色/亮色主题、流式输出、Mermaid 图表 |

## 🚀 快速开始

### 前置条件

- Python 3.10+
- [Ollama](https://ollama.ai/)（本地 LLM 运行环境）
- 至少 8GB 显存（推荐 16GB）

### 1. 安装 Ollama 模型

```bash
# 拉取嵌入模型
ollama pull bge-m3

# 创建对话模型（基于 Qwen3.5）
ollama create qwen35-campus -f Modelfile.qwen35
```

### 2. 克隆并安装

```bash
git clone https://github.com/Men-Zer/CampusAgent.git
cd CampusAgent

# 虚拟环境（推荐）
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 配置并启动

```bash
cp .env.example .env
# 或 Windows: copy .env.example .env

python main.py
```

打开浏览器访问 [http://localhost:8000](http://localhost:8000)

### 🐳 Docker 启动（推荐）

```bash
docker compose up -d
# 访问 http://localhost:8000
```

## 🏗️ 技术架构

```
用户输入 → ReAct Agent (LangGraph)
                ├── 知识库检索（双路召回 + bge-reranker 重排）
                ├── 联网搜索（百度 → Tavily 降级）
                ├── 网页抓取
                └── 时钟工具
                    ↓
            LLM 生成回答（Qwen3.5 / Ollama）
                    ↓
            流式输出 SSE → 前端 UI
```

### 关键技术指标

| 指标 | 数值 |
|------|------|
| 知识库 Recall@5 | 89%（纯向量 68% → 双路+重排 89%，+21pp） |
| 搜索引擎覆盖率 | 99%+（百度 + Tavily 双引擎降级） |
| API 响应 P50 | 5.2s |
| 短期记忆 | 10 轮对话 |
| 长程记忆 | ChromaDB 向量持久化 |

## 🔒 安全特性

- SSRF 防护：双重 IP 校验，拒绝内网访问
- 路径穿越防护：用户文件路径严格校验
- 文件上传：类型白名单 + 20MB 上限
- Prompt 注入缓解：结构化指令隔离

## 🤝 参与贡献

欢迎任何形式的贡献！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

**如果这个项目对你有帮助，欢迎给个 ⭐ ⭐ ⭐ ⭐ ⭐ — 你的支持是作者持续更新的动力！**
