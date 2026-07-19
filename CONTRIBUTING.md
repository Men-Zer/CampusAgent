# 贡献指南

感谢你对 AgenticRAG 的兴趣！欢迎任何形式的贡献。

## 贡献方式

### 🐛 报告 Bug

1. 先搜索 [Issues](https://github.com/Men-Zer/AgenticRAG/issues) 看是否已被报告
2. 如果不存在，[创建新 Issue](https://github.com/Men-Zer/AgenticRAG/issues/new?template=bug_report.md)
   - 清晰描述问题
   - 提供复现步骤
   - 附上环境信息（OS、Python 版本、Ollama 版本等）

### 💡 提交功能建议

[创建 Feature Request](https://github.com/Men-Zer/AgenticRAG/issues/new?template=feature_request.md)
   - 说明你想要的功能
   - 描述使用场景
   - 如果可能，提供实现思路

### 🛠️ 提交代码

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'feat: add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 提交 Pull Request

### 📖 改进文档

修正错别字、完善文档、补充示例都欢迎。

## 开发指南

### 环境准备

```bash
# 假设你已经按照 README 完成了基础安装

# 安装开发依赖
pip install pytest flake8
```

### 代码规范

- Python 代码遵循 PEP 8
- 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/)
- 新增功能需包含测试

### 运行测试

```bash
pytest tests/ -v
```

## 行为准则

请保持友善、尊重、建设性的交流氛围。
