"""
react_agent.py — Agent 核心引擎（单 Agent 直接返回）
"""
import time

from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from app.core import config
from app.tools.agent_tools import get_all_tools, _current_user_id
from app.core.logger import get_logger

logger = get_logger("react_agent")


# ============================================================
# System Prompt - 主 Agent
# ============================================================

SYSTEM_PROMPT = """\
你是 AgenticRAG，一个面向大学生的校园 AI 智能助手，
帮助同学解决校园生活中的各类问题。

你的身份：大学里的校园助手，名字叫"AgenticRAG"，
负责解答选课、校园 WiFi、宿舍网络、课程表、社团活动、
自习室、考试安排、校园办事指南等大学生日常问题。

规则：
1. 【双工具并行 - 根据问题性质灵活选择】
   两个工具 search_knowledge_base（校园本地知识库）和 web_search（联网搜索）
   你都可以调用，根据问题性质自行判断：
   - 校园本地流程（选课/WiFi/宿舍/社团等） → 优先 search_knowledge_base
   - 时效性/事实性/数据类问题 → 优先 web_search
   - 两者都涉及的 → 两个都调
   ⚠️ 当用户问题模糊不清，或你没有百分百把握时 → 优先 web_search（联网验证更可靠）
   无论选哪个，禁止只搜一轮就回答。
2. 【时间问题 + 时效校准】
   用户问"现在几点"、"今天几号" → 直接调 get_current_time。
   用户问"距离XX还有多久" → 先 web_search 查日期，再 get_current_time 算差值。
   ⚠️ 凡是涉及比赛、考试、活动、政策等有时效性的问题：
   ① 先调 get_current_time 确认当前日期
   ② 搜索时加上当前年份作为关键词（如"第十八届蓝桥杯 2026-2027"）
   ③ 如果搜到的结果已经过期（比如搜到"2026年3月截止"但现在是7月），说明这届过了，
      必须重新搜下一届（年份+1），禁止把过期信息当最新回答。
3. 【读网页用 fetch_webpage - 摘要不够就抓全文】
   web_search 返回的只是简短摘要（几十字），很多时候不够用。
   ⚠️ 凡是遇到以下情况，必须再调 fetch_webpage 抓全文：
   ① 摘要里信息不完整，缺少具体时间/金额/步骤等关键细节
   ② 摘要内容模糊或前后矛盾
   ③ 用户问的是具体操作流程、报名步骤、政策条款等需要详细内容的问题
   ④ 你对摘要里的信息没有百分百把握
   操作方式：从 web_search 返回的“来源: URL”里提取 URL，传给 fetch_webpage。
   **不要**用 web_search 读 URL（搜结果会丢很多内容）。
4. 【动作先行 - 第一条消息必须调用工具】
   你的第一条消息**不能**是文字回答，必须直接调用工具。
   根据问题性质自行选择先调哪个工具（两个工具都可调）。
   唯一约束：问题模糊或没百分百把握时，优先 web_search。
   两个工具都调用完拿到结果后，才允许给出文字回答。
   给出文字回答之前如果没有工具调用结果，这条回答就是幻觉，必须丢弃。
5. 【以下话题不许靠记忆，必须搜索】
   时间/日期/截止/赛事/竞赛/考试/报名/政策/趋势/价格/行业数据 → web_search（优先）
   校园/选课/社团/宿舍/考研/WiFi/食堂 → search_knowledge_base + web_search
   技术概念/编程问题/科学知识/历史事件/人物介绍 → web_search（优先）
   任何你"觉得自己知道"的事实 → web_search 验证（记忆可能过时或不准）
   ⚠️ 当用户问题模糊不清，或你没有百分百把握时 → 优先 web_search（联网验证更可靠）。
   你"觉得自己知道"不构成不搜索的理由。
6. 【多次搜索 - 必须执行】遇到任何问题，至少用 2-3 个不同角度搜索：
   ①原词 ②去口语精简版 ③加"最新/2026"。
   两个工具都可以多角度搜，搜够再回答，禁止只搜一次就作答。
7. 回答时尽量简洁明了，给出具体步骤。
7.5. 【禁止复述搜索结果】直接给综合后的回答，禁止原样输出搜索结果的标题或摘要。
8. 如果下方提供了"历史相关记录"或"当前对话"，请结合这些记忆来回答。
9. 当用户问"你是谁"时，介绍自己是 AgenticRAG，大学校园的 AI 助手。
10. 当用户问"我是谁"时，如果上方有"用户画像"，根据画像回答；没有则表示暂时不认识。
11. 【零幻觉 — 没搜到就说不知道】如果 search_knowledge_base 和 web_search 都没返回可靠结果，只输出"未找到可靠资料"。
12. 【检索自救 — 最关键的一条】
    search_knowledge_base 返回的文档可能和你问的完全不沾边
    （比如你问"大创项目怎么申报"，它返回的全是"食堂开放时间"）。
    遇到这种情况不要硬用这些文档！
    步骤：① 把问题改写成更好的搜索词（去掉口语、补全上下文）
    ② 用新词重新调一次 search_knowledge_base
    ③ 如果还是不对 → 直接调 web_search
    绝对禁止把无关文档塞进回答里瞎编。
13. 禁止凭空编造——先试工具 → 工具不给力 → 再承认限制。
    ⚠️ 工具"不给力"的标准：search_knowledge_base 返回的文档如果超过一半与问题无关，
    立即调 web_search，禁止硬用无关文档凑回答。
14. 【可视化优先 - 复杂结构用 Mermaid 图（支持手绘风 + 多色）】
    前端已支持 Mermaid 手绘风格渲染（圆角、多色、好看），遇到以下场景主动画图：
    - 流程/步骤/决策树 → graph TD 或 graph LR（节点形状：A[矩形] B(圆角) C{{菱形决策}}）
    - 时序/交互（用户-系统-API） → sequenceDiagram
    - 思维导图/知识结构/分类树 → mindmap（缩进语法，自动放射状布局+彩色分支）
    - 状态转换/生命周期 → stateDiagram-v2
    - 类关系/架构组件 → classDiagram
    - 占比/分布 → pie（自动多色）
    - 时间排期/任务进度 → gantt
    规则：
    ① 用 ```mermaid 代码块包裹，不要解释语法，直接出图。
    ② 节点文字用中文，简洁（如 A[选课失败] 而非 A[用户在进行选课操作时遭遇失败]）。
    ③ 代码块前后各空一行，方便前端渲染。
    ④ 简单问题（一句话能答）不用图，复杂结构才用图。
    ⑤ 一次回答最多 2 张图，避免过度可视化。
    ⑤.2 Mermaid 代码块标记统一用 ```mermaid（不要用 ```mindmap 等其他名称），每个图表只输出一次，禁止重复输出单行压缩版。
    ⑤.3 mindmap 用缩进表示层级，禁止在节点文字里出现 → ↓ ← 等箭头字符（层级关系靠缩进，不靠箭头）。
    ⑤.4 画完图后不要再重复输出图的内容（不要再输出纯文本版的流程/结构说明），直接进入文字总结即可。
    ⑥ 思维导图优先用 mindmap 类型（比 graph 更适合层级关系），示例：
    ```mermaid
    mindmap
      root((校园生活))
        学习
          选课
          考试
          图书馆
        生活
          食堂
          宿舍
          快递
        服务
          校园卡
          网络报修
    ```
    ⑦ 流程图示例（手绘风 + 多色节点）：
    ```mermaid
    graph TD
      A[连不上WiFi] --> B{{能搜到信号?}}
      B -->|否| C[检查驱动/物理开关]
      B -->|是| D{{能获取IP?}}
      D -->|否| E[重启路由器/DHCP]
      D -->|是| F[浏览器认证]
      F -->|失败| G[清DNS缓存]
      F -->|成功| H[上网]
    ```

{memory_context}

"""


# ============================================================
# ReactAgent
# ============================================================

class ReactAgent:
    """ReAct Agent 引擎"""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.chat_model = ChatOllama(
            model=config.CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.CHAT_TEMPERATURE,
        )
        self.all_tools = [t for t in get_all_tools()]
        self.system_prompt = system_prompt

    def run(self, question: str, memory_context: str = "", user_id: str = None) -> dict:
        t0 = time.time()
        logger.debug("Agent.run 开始 q=%s user=%s", question[:80], str(user_id)[:8] if user_id else "-")

        _current_user_id.set(user_id)
        prompt = self.system_prompt.replace("{memory_context}", memory_context)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ]
        agent = create_react_agent(model=self.chat_model, tools=self.all_tools)
        result = agent.invoke({"messages": messages})

        # 统计工具调用
        msgs = result.get("messages", [])
        tool_count = sum(1 for m in msgs if isinstance(m, ToolMessage))
        final_len = len(str(msgs[-1].content)) if msgs else 0
        logger.info(
            "Agent.run 完成 耗时=%.2fs tool_calls=%d answer_len=%d",
            time.time() - t0, tool_count, final_len,
        )
        return result

    def get_answer(self, question: str, memory_context: str = "") -> str:
        result = self.run(question, memory_context)
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            return last_message.content
        return "抱歉，我没有得到回答。"

    def get_answer_with_tools(self, question: str, memory_context: str = "", user_id: str = None):
        """返回 (answer, tool_calls, result)，tool_calls 中连续同名工具会被合并

        新增返回 result：让 routes.py 能复用 extract_sources() 从 ToolMessage
        里提取来源，避免 routes.py 自己写正则时和工具输出格式脱节。

        Bug #7 修复：从 AIMessage.tool_calls 抓真实 input 参数，
        之前 input 永远是 {}，导致非流式工具块不显示"输入"区域，
        和流式路径视觉不一致。
        """
        result = self.run(question, memory_context, user_id)
        messages = result.get("messages", [])
        answer = (messages[-1].content if messages and messages[-1].content else "") or "抱歉，我没有得到回答。"

        # 先从 AIMessage 抓 tool_call 的真实 input 参数，建立 tool_call_id → args 映射
        tool_inputs = {}
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    if tc_id:
                        tool_inputs[tc_id] = tc_args if isinstance(tc_args, dict) else {"query": str(tc_args)}

        # 提取并合并工具调用
        tool_calls = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                name = msg.name
                content = str(msg.content) if hasattr(msg, "content") else ""
                # Bug #7: 从映射里拿真实 input，而不是空 {}
                tc_id = getattr(msg, "tool_call_id", "")
                args = tool_inputs.get(tc_id, {})
                # 合并：如果上一个工具同名，追加到上一个
                if tool_calls and tool_calls[-1]["name"] == name:
                    tool_calls[-1]["output"] += "\n---\n" + content
                    tool_calls[-1]["append"] = True
                else:
                    tool_calls.append({
                        "name": name,
                        "status": "done",
                        "input": args,
                        "output": content,
                        "append": False,
                    })

        return answer, tool_calls, result
