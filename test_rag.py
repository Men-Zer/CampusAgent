# -*- coding: utf-8 -*-
"""test_rag.py - RAG 三层测试: 冒烟 / reranker对比 / 召回率"""
import os, sys, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from app.services.vector_store import VectorStore
from app.services.reranker import rerank, is_degraded
from app.tools.agent_tools import search_knowledge_base, _get_general_store
from app.core import config

# 标注已根据实际文件内容核实修正:
# - 校园卡丢失补办 -> campus_id_card_guide.md (含"挂失+补办"章节)
# - 学校有校招吗 -> campus_internship_guide.md (resume_interview 只讲简历面试无校招)
# - 学校放假时间 -> campus_spring_festival_travel.md (全库无校历,最接近寒暑假返乡指南)
# - 宿舍网络 -> campus_wifi_network.md (dorm_life 讲宿舍生活不讲网络配置)
GROUND_TRUTH = [
    {"q": "校园 WiFi 连不上怎么办", "expect": ["campus_wifi_network.md"]},
    {"q": "选课系统进不去怎么办", "expect": ["campus_course_selection.md"]},
    {"q": "宿舍网络怎么配置", "expect": ["campus_wifi_network.md"]},
    {"q": "校园卡丢了怎么补办", "expect": ["campus_id_card_guide.md"]},
    {"q": "食堂开放时间是什么时候", "expect": ["campus_canteen_food.md"]},
    {"q": "考研什么时候开始报名", "expect": ["campus_kaoyan_strategy.md", "campus_postgraduate_timeline.md"]},
    {"q": "四六级考试时间是什么时候", "expect": ["campus_exam_study.md"]},
    {"q": "图书馆怎么借书", "expect": ["campus_library_advanced.md"]},
    {"q": "怎么申请奖学金", "expect": ["campus_scholarship_finance.md"]},
    {"q": "毕业论文查重怎么过", "expect": ["campus_thesis_defense.md"]},
    {"q": "怎么加入社团", "expect": ["campus_club_social.md"]},
    {"q": "学校有哪些社团可以参加", "expect": ["campus_club_social.md"]},
    {"q": "大二找实习需要准备什么", "expect": ["campus_internship_guide.md"]},
    {"q": "学校有校招吗", "expect": ["campus_internship_guide.md"]},
    {"q": "学校放假时间怎么查", "expect": ["campus_spring_festival_travel.md"]},
]

def extract_source_filename(doc):
    source = doc.metadata.get("source", "")
    return os.path.basename(source) if source else ""

def extract_titles_from_tool_output(output):
    return re.findall(r"来自《(.+?)》", output)

def filename_to_clean_title(fname):
    stem = os.path.splitext(fname)[0]
    return stem.replace("campus_", "").replace("_", " ")

def test_smoke():
    print("=" * 60)
    print("层次 1: 冒烟测试 - 验证检索流程跑通")
    print("=" * 60)
    q = "校园 WiFi 连不上怎么办"
    print(f"问题: {q}")
    print("调用 search_knowledge_base() ...")
    t0 = time.time()
    result = search_knowledge_base(q)
    dt = time.time() - t0
    titles = extract_titles_from_tool_output(result)
    print(f"耗时: {dt:.2f}s")
    print(f"返回文档数: {len(titles)}")
    for t in titles:
        print(f"  - {t}")
    print(f"reranker 降级模式: {is_degraded()}")
    if titles:
        print("[结论] 检索流程正常, 返回了文档")
    else:
        print("[警告] 未返回任何文档, 检查向量库或 ollama")
    print()


def test_reranker_compare():
    print("=" * 60)
    print("层次 2: reranker 对比 - 验证 reranker 改变了排序")
    print("=" * 60)
    q = "考研什么时候开始报名"
    print(f"问题: {q}")
    store = _get_general_store()
    t0 = time.time()
    vector_docs = store.search(q, k=20)
    dt1 = time.time() - t0
    print(f"\n[仅向量检索] 耗时 {dt1:.2f}s, 召回 {len(vector_docs)} 条 (前5):")
    for i, d in enumerate(vector_docs[:5], 1):
        fname = extract_source_filename(d)
        print(f"  {i}. {fname} | {d.page_content[:50]}...")
    if vector_docs:
        t0 = time.time()
        reranked = rerank(q, vector_docs, top_k=5)
        dt2 = time.time() - t0
        print(f"\n[reranker 重排] 耗时 {dt2:.2f}s, 保留 {len(reranked)} 条:")
        for i, d in enumerate(reranked, 1):
            fname = extract_source_filename(d)
            print(f"  {i}. {fname} | {d.page_content[:50]}...")
        vector_files = [extract_source_filename(d) for d in vector_docs[:5]]
        reranked_files = [extract_source_filename(d) for d in reranked]
        if vector_files != reranked_files:
            print("\n[结论] reranker 改变了排序! 它在起作用")
        else:
            print("\n[结论] 顺序未变 (可能原排序已最优)")
    else:
        print("\n[跳过] 向量检索无结果")
    print()


def test_recall():
    print("=" * 60)
    print("层次 3: 召回率 Recall@4 (完整流程: 向量+BM25+reranker)")
    print("=" * 60)
    total = len(GROUND_TRUTH)
    hit = 0
    details = []
    for i, case in enumerate(GROUND_TRUTH, 1):
        q = case["q"]
        expect = case["expect"]
        try:
            output = search_knowledge_base(q)
            titles = extract_titles_from_tool_output(output)
        except Exception as e:
            titles = []
            output = f"ERROR: {e}"
        expect_clean = [filename_to_clean_title(f) for f in expect]
        matched = any(ec in " | ".join(titles) for ec in expect_clean)
        if not matched:
            for f in expect:
                stem = os.path.splitext(f)[0]
                if stem in output:
                    matched = True
                    break
        if matched:
            hit += 1
            status = "PASS"
        else:
            status = "FAIL"
        details.append({"q": q, "expect": expect, "got": titles, "hit": matched})
        print(f"[{i}/{total}] {status} | {q[:35]}")
        print(f"         期望: {expect}")
        print(f"         实际: {titles[:3] if titles else '(空)'}")
    recall = hit / total * 100
    print()
    print("=" * 60)
    print(f"召回率 Recall@4: {recall:.1f}% ({hit}/{total})")
    print("=" * 60)
    fails = [d for d in details if not d["hit"]]
    if fails:
        print(f"\n失败用例 ({len(fails)} 条):")
        for f in fails:
            print(f"  Q: {f['q']}")
            print(f"    期望: {f['expect']}")
            print(f"    实际: {f['got']}")


if __name__ == "__main__":
    test_smoke()
    test_reranker_compare()
    test_recall()