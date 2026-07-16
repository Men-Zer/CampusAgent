"""
CampusAgent 自动评测脚本
用法：python eval.py
输出：准确率 / 召回率 / 平均响应时间 / 逐条详情
"""
import json
import time
import sys
import os
import urllib.request
import urllib.error

# ---------- 配置 ----------
API_URL = "http://127.0.0.1:8000/chat"
REQUEST_TIMEOUT = 60  # 单次请求最长等待秒数

# ---------- 评测数据集（20 条标注 QA）----------
# 每条：question / expected_keywords（至少命中 1 个即算通过）
TEST_CASES = [
    # ===== 校园生活 =====
    {
        "id": 1,
        "question": "校园 WiFi 连不上怎么办",
        "expected_keywords": ["忘记网络", "重新连接", "WiFi", "密码", "信号"],
        "category": "校园生活"
    },
    {
        "id": 2,
        "question": "选课系统进不去怎么办",
        "expected_keywords": ["选课", "系统", "刷新", "浏览器", "缓存", "VPN"],
        "category": "选课"
    },
    {
        "id": 3,
        "question": "宿舍网络怎么配置",
        "expected_keywords": ["网络", "路由", "IP", "DNS", "配置"],
        "category": "校园生活"
    },
    {
        "id": 4,
        "question": "校园卡丢了怎么补办",
        "expected_keywords": ["校园卡", "补办", "挂失", "一卡通", "卡务"],
        "category": "校园生活"
    },
    {
        "id": 5,
        "question": "食堂开放时间是什么时候",
        "expected_keywords": ["食堂", "开放", "时间", "早餐", "午餐", "晚餐"],
        "category": "校园生活"
    },
    # ===== 学习考试 =====
    {
        "id": 6,
        "question": "考研什么时候开始报名",
        "expected_keywords": ["考研", "报名", "时间", "研究生", "初试"],
        "category": "考研"
    },
    {
        "id": 7,
        "question": "四六级考试时间是什么时候",
        "expected_keywords": ["四六级", "考试", "时间", "英语", "CET"],
        "category": "考试"
    },
    {
        "id": 8,
        "question": "图书馆怎么借书",
        "expected_keywords": ["图书馆", "借书", "借阅", "还书", "图书"],
        "category": "学习"
    },
    {
        "id": 9,
        "question": "怎么申请奖学金",
        "expected_keywords": ["奖学金", "申请", "条件", "成绩", "材料"],
        "category": "学习"
    },
    {
        "id": 10,
        "question": "毕业论文查重怎么过",
        "expected_keywords": ["论文", "查重", "重复率", "引用", "格式"],
        "category": "学习"
    },
    # ===== 社团活动 =====
    {
        "id": 11,
        "question": "怎么加入社团",
        "expected_keywords": ["社团", "加入", "报名", "招新", "面试"],
        "category": "社团"
    },
    {
        "id": 12,
        "question": "学校有哪些社团可以参加",
        "expected_keywords": ["社团", "类型", "兴趣", "运动", "文艺", "科技"],
        "category": "社团"
    },
    # ===== 系统操作 =====
    # ===== 实习就业 =====
    {
        "id": 16,
        "question": "大二找实习需要准备什么",
        "expected_keywords": ["实习", "准备", "简历", "面试", "技能", "经验"],
        "category": "实习"
    },
    {
        "id": 17,
        "question": "学校有校招吗",
        "expected_keywords": ["校招", "招聘", "企业", "春招", "秋招", "就业"],
        "category": "实习"
    },
    # ===== 通知公告 =====
    {
        "id": 19,
        "question": "学校放假时间怎么查",
        "expected_keywords": ["放假", "假期", "校历", "时间", "寒假", "暑假"],
        "category": "教务"
    },
    # ===== 混合意图 =====
    {
        "id": 20,
        "question": "选课和考研冲突了怎么办",
        "expected_keywords": ["选课", "考研", "冲突", "时间", "安排", "调整"],
        "category": "混合"
    },
]


def call_api(question: str) -> dict:
    """单次 API 调用，返回 {success, answer, elapsed_ms, error}"""
    start = time.time()
    try:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps({"message": question, "session_id": "eval_session"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
        body = json.loads(resp.read().decode())
        elapsed_ms = (time.time() - start) * 1000
        return {"success": True, "answer": body.get("answer", ""), "elapsed_ms": elapsed_ms, "error": None}
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return {"success": False, "answer": "", "elapsed_ms": elapsed_ms, "error": str(e)}


def check_answer(answer: str, keywords: list[str]) -> bool:
    """检查回答是否包含至少 1 个期望关键词"""
    lower = answer.lower()
    return any(kw.lower() in lower for kw in keywords)


def main():
    print("=" * 60)
    print("CampusAgent 自动评测")
    print(f"评测用例数: {len(TEST_CASES)}")
    print(f"API 地址: {API_URL}")
    print("=" * 60)

    success_count = 0
    fail_details = []
    total_elapsed = 0.0

    for i, case in enumerate(TEST_CASES, 1):
        qid = case["id"]
        question = case["question"]
        keywords = case["expected_keywords"]

        print(f"\n[{i}/{len(TEST_CASES)}] Q{qid}: {question[:40]}...", end=" ", flush=True)

        result = call_api(question)
        total_elapsed += result["elapsed_ms"]

        if not result["success"]:
            print(f"❌ API错误 ({result['elapsed_ms']:.0f}ms): {result['error'][:60]}")
            fail_details.append({
                "id": qid, "question": question, "category": case["category"],
                "reason": f"API错误: {result['error'][:80]}",
                "elapsed_ms": result["elapsed_ms"],
            })
            continue

        passed = check_answer(result["answer"], keywords)
        if passed:
            print(f"✅ PASS ({result['elapsed_ms']:.0f}ms)")
            success_count += 1
        else:
            snippet = result["answer"][:80].replace("\n", " ")
            print(f"❌ FAIL ({result['elapsed_ms']:.0f}ms) | 预期关键词: {keywords[:3]} | 回答片段: {snippet}")
            fail_details.append({
                "id": qid, "question": question, "category": case["category"],
                "expected_keywords": keywords,
                "answer_snippet": result["answer"][:200],
                "elapsed_ms": result["elapsed_ms"],
            })

    # ── 统计 ──
    total = len(TEST_CASES)
    accuracy = success_count / total * 100
    avg_elapsed = total_elapsed / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("评测结果")
    print("=" * 60)
    print(f"  总计:    {total} 条")
    print(f"  通过:    {success_count} 条")
    print(f"  失败:    {total - success_count} 条")
    print(f"  准确率:  {accuracy:.1f}%")
    print(f"  平均响应: {avg_elapsed:.0f}ms")

    # ── 分类统计 ──
    by_cat = {}
    for case in TEST_CASES:
        cat = case["category"]
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "passed": 0}
        by_cat[cat]["total"] += 1
    for fail in fail_details:
        cat = TEST_CASES[fail["id"] - 1]["category"]
        by_cat[cat]["passed"] = by_cat[cat]["total"]  # 先全部算通过
    # 重新算
    for cat in by_cat:
        cat_fails = [f for f in fail_details if f["id"] in 
                     [c["id"] for c in TEST_CASES if c["category"] == cat]]
        # 简化：按 category 统计
        total_cat = len([c for c in TEST_CASES if c["category"] == cat])
        failed_cat = len([f for f in fail_details if 
                          any(c["id"] == f["id"] and c["category"] == cat for c in TEST_CASES)])
        by_cat[cat] = {"total": total_cat, "passed": total_cat - failed_cat}

    print("\n  分类准确率:")
    for cat, stats in sorted(by_cat.items()):
        pct = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"    {cat:6s}  {bar}  {pct:.0f}% ({stats['passed']}/{stats['total']})")

    # ── 失败详情 ──
    if fail_details:
        print(f"\n  失败详情 ({len(fail_details)} 条):")
        for f in fail_details:
            # 找 category
            for c in TEST_CASES:
                if c["id"] == f["id"]:
                    cat = c["category"]
                    break
            print(f"    Q{f['id']} [{cat}] {f['question'][:50]}")
            if "answer_snippet" in f:
                print(f"         预期: {f.get('expected_keywords', [])[:3]}")
                print(f"         实际: {f['answer_snippet'][:100]}")
            else:
                print(f"         {f['reason']}")
            print()

    # ── 保存 JSON 报告 ──
    report = {
        "total": total,
        "passed": success_count,
        "failed": total - success_count,
        "accuracy": round(accuracy, 1),
        "avg_elapsed_ms": round(avg_elapsed, 0),
        "failures": fail_details,
    }
    out_path = os.path.join(os.path.dirname(__file__), "eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  详细报告已保存: {out_path}")

    print("=" * 60)
    return 0 if accuracy >= 60 else 1


if __name__ == "__main__":
    sys.exit(main())
