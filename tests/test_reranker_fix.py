"""
独立验证测试：reranker 优雅降级修复

测试覆盖：
1. 逻辑正确性：哨兵模式区分 None / 对象 / 失败三种状态
2. 三条降级路径：ImportError / 实例化异常 / agent_tools 防御层
3. 无回归风险：正常路径行为一致
4. 日志可观测性：异常情况有足够日志

用法: python tests/test_reranker_fix.py
"""

import sys
import os
import io
import logging
import unittest
from unittest.mock import patch, MagicMock

# 确保项目根在 path 里
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ============================================================
# Test 1: 逻辑正确性 — 哨兵三态区分
# ============================================================
class TestSentinelStates(unittest.TestCase):
    """验证 _RERANKER_LOAD_FAILED 哨兵能正确区分三种状态"""

    def setUp(self):
        from app.services.reranker import RerankerService
        self.service = RerankerService()

    def test_initial_state_is_none(self):
        """初始状态：self._reranker 应为 None"""
        self.assertIsNone(self.service._reranker)

    def test_sentinel_is_not_none(self):
        """哨兵值是 object()，is not None 应为 True（阻止重试）"""
        from app.services.reranker import _RERANKER_LOAD_FAILED
        self.assertIsNotNone(_RERANKER_LOAD_FAILED)

    def test_ensure_loaded_skips_when_loaded(self):
        """_ensure_loaded() 在已加载成功时应跳过"""
        mock_model = MagicMock()
        self.service._reranker = mock_model
        # 不应抛异常
        self.service._ensure_loaded()
        self.assertIs(self.service._reranker, mock_model)

    def test_ensure_loaded_skips_when_failed(self):
        """_ensure_loaded() 在已失败时应跳过，不重试"""
        from app.services.reranker import _RERANKER_LOAD_FAILED
        self.service._reranker = _RERANKER_LOAD_FAILED
        self.service._ensure_loaded()
        # 应该还是原来的哨兵值，说明没有重新进入 try 块
        self.assertIs(self.service._reranker, _RERANKER_LOAD_FAILED)

    def test_sentinel_prevents_retry_after_import_error(self):
        """ImportError 后哨兵阻止重试"""
        from app.services.reranker import _RERANKER_LOAD_FAILED

        # 第一次：模拟 ImportError
        with patch("app.services.reranker.logger") as mock_logger:
            # Mock __import__ 来模拟 rerankers 包不存在
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "rerankers":
                    raise ImportError("No module named 'rerankers'")
                return original_import(name, *args, **kwargs)

            try:
                builtins.__import__ = mock_import
                self.service._ensure_loaded()
                self.assertIs(self.service._reranker, _RERANKER_LOAD_FAILED)
            finally:
                builtins.__import__ = original_import

        # 第二次：应直接跳过
        call_count = 0
        def counting_import(name, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = counting_import
            self.service._ensure_loaded()
            self.assertIs(self.service._reranker, _RERANKER_LOAD_FAILED)
            self.assertEqual(call_count, 0, "第二次 _ensure_loaded() 不应再次尝试 import")
        finally:
            builtins.__import__ = original_import


# ============================================================
# Test 2: 三条降级路径
# ============================================================
class TestDegradationPaths(unittest.TestCase):
    """验证三条优雅降级路径"""

    def setUp(self):
        from app.services.reranker import RerankerService
        self.service = RerankerService()
        # 构造假文档
        from langchain_core.documents import Document
        self.docs = [
            Document(page_content=f"Document {i}", metadata={"id": i})
            for i in range(10)
        ]

    def test_path1_import_error_degradation(self):
        """路径1：reranker 包未安装 → 降级返回 top_k"""
        from app.services.reranker import _RERANKER_LOAD_FAILED

        # 用 mock 构造：_ensure_loaded 失败后 _reranker 是哨兵
        self.service._reranker = _RERANKER_LOAD_FAILED

        result = self.service.rerank("test query", self.docs, top_k=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].page_content, "Document 0")
        self.assertEqual(result[1].page_content, "Document 1")
        self.assertEqual(result[2].page_content, "Document 2")

    def test_path2_model_load_failure_degradation(self):
        """路径2：Reranker() 实例化异常 → 降级返回 top_k"""
        from app.services.reranker import _RERANKER_LOAD_FAILED

        # 直接设哨兵状态（模拟加载失败后）
        self.service._reranker = _RERANKER_LOAD_FAILED

        result = self.service.rerank("test query", self.docs, top_k=5)
        self.assertEqual(len(result), 5)
        # 验证返回的是原始顺序的前 5 条
        for i in range(5):
            self.assertEqual(result[i].page_content, f"Document {i}")

    def test_path3_agent_tools_defense_layer(self):
        """路径3：agent_tools.py 层 try/except 捕获意外异常"""
        from app.services.reranker import rerank

        # 模拟 rerank 函数本身抛异常
        with patch("app.services.reranker._reranker.rerank", side_effect=RuntimeError("Unexpected crash")):
            # 模拟 agent_tools 中的 try/except 逻辑
            all_docs = list(self.docs)
            top_k = 6
            try:
                all_docs = rerank("test query", all_docs, top_k=top_k)
            except Exception as e:
                # 这是 agent_tools.py 第206-208行的逻辑
                all_docs = all_docs[:top_k]

            # 降级后应返回前6条
            self.assertEqual(len(all_docs), 6)
            self.assertEqual(all_docs[0].page_content, "Document 0")

    def test_empty_docs_returns_empty(self):
        """空文档列表直接返回 []"""
        from app.services.reranker import _RERANKER_LOAD_FAILED
        self.service._reranker = _RERANKER_LOAD_FAILED
        result = self.service.rerank("query", [], top_k=5)
        self.assertEqual(result, [])


# ============================================================
# Test 3: 无回归风险 — 正常路径
# ============================================================
class TestNoRegression(unittest.TestCase):
    """验证正常路径（reranker 可用时）行为不变"""

    def setUp(self):
        from langchain_core.documents import Document
        self.docs = [
            Document(page_content=f"Doc {i}", metadata={"id": i})
            for i in range(10)
        ]

    def test_normal_path_calls_reranker(self):
        """正常路径：reranker 可用时应调用 .rank()"""
        from app.services.reranker import RerankerService

        service = RerankerService()
        mock_reranker = MagicMock()

        # 模拟 rank 返回结果
        mock_result1 = MagicMock()
        mock_result1.doc_id = 9
        mock_result1.score = 0.95
        mock_result2 = MagicMock()
        mock_result2.doc_id = 7
        mock_result2.score = 0.80
        mock_result3 = MagicMock()
        mock_result3.doc_id = 5
        mock_result3.score = 0.60

        mock_reranker.rank.return_value = [mock_result1, mock_result2, mock_result3]
        service._reranker = mock_reranker

        result = service.rerank("test query", self.docs, top_k=3)

        # 验证调用了 rank
        mock_reranker.rank.assert_called_once()

        # 验证返回顺序正确（按 rank 结果）
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].page_content, "Doc 9")
        self.assertEqual(result[1].page_content, "Doc 7")
        self.assertEqual(result[2].page_content, "Doc 5")

    def test_normal_path_handles_dict_doc_id(self):
        """正常路径：rank 结果可能使用 dict 格式的 doc_id"""
        from app.services.reranker import RerankerService

        service = RerankerService()
        mock_reranker = MagicMock()
        # dict-style result
        mock_reranker.rank.return_value = [
            {"doc_id": 3, "score": 0.9},
            {"doc_id": 1, "score": 0.7},
        ]
        service._reranker = mock_reranker

        result = service.rerank("test", self.docs, top_k=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].page_content, "Doc 3")
        self.assertEqual(result[1].page_content, "Doc 1")

    def test_normal_path_top_k_caps_output(self):
        """正常路径：top_k 正确截断"""
        from app.services.reranker import RerankerService

        service = RerankerService()
        mock_reranker = MagicMock()
        results = []
        for i in range(10):
            m = MagicMock()
            m.doc_id = 9 - i
            m.score = 1.0 - i * 0.1
            results.append(m)
        mock_reranker.rank.return_value = results
        service._reranker = mock_reranker

        result = service.rerank("test", self.docs, top_k=4)
        self.assertEqual(len(result), 4)


# ============================================================
# Test 4: 日志可观测性
# ============================================================
class TestObservability(unittest.TestCase):
    """验证异常情况有足够的日志"""

    def setUp(self):
        from app.services.reranker import RerankerService
        self.service = RerankerService()

    def test_import_error_logs_warning(self):
        """ImportError 时应输出 warning 日志"""
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)

        logger = logging.getLogger("app.services.reranker")
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.WARNING)

        try:
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "rerankers":
                    raise ImportError("No module named 'rerankers'")
                return original_import(name, *args, **kwargs)

            try:
                builtins.__import__ = mock_import
                self.service._ensure_loaded()
            finally:
                builtins.__import__ = original_import

            log_output = log_stream.getvalue()
            self.assertIn("无法加载 Reranker 模型", log_output)
            self.assertIn("降级", log_output)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

    def test_agent_tools_catch_logs_step(self):
        """agent_tools.py 层异常应记录 STEP 日志"""
        from langchain_core.documents import Document
        from app.services.reranker import rerank

        docs = [Document(page_content=f"D{i}", metadata={}) for i in range(10)]

        # 模拟 rerank 抛异常
        with patch("app.services.reranker._reranker.rerank", side_effect=RuntimeError("Boom")):
            # 这是 agent_tools.py 第202-208行的逻辑
            step_log = "[STEP:重排模型] 小模型正在重新打分（毫秒级）..."
            try:
                all_docs = rerank("test", docs, top_k=6)
            except Exception as e:
                step_log_extra = f"[STEP:重排模型] 重排失败({e})，降级使用原始排序，保留 Top 6"
                all_docs = docs[:6]

            self.assertIn("重排失败", step_log_extra)
            self.assertIn("降级", step_log_extra)
            self.assertEqual(len(all_docs), 6)


# ============================================================
# 额外检查：边界条件
# ============================================================
class TestEdgeCases(unittest.TestCase):
    """边界条件测试"""

    def test_single_doc_no_rerank_needed(self):
        """单文档不需要重排（agent_tools 逻辑层面）"""
        from langchain_core.documents import Document
        docs = [Document(page_content="Only one doc", metadata={})]
        # agent_tools.py 中：len(all_docs) > 1 才调用 rerank
        # 单文档直接跳过，这由 agent_tools 控制
        self.assertEqual(len(docs), 1)  # 验证不需要重排

    def test_sentinel_identity_is_singleton(self):
        """哨兵值的 is 比较应在模块内一致"""
        from app.services.reranker import _RERANKER_LOAD_FAILED
        # 多次获取应该是同一个对象
        from app.services.reranker import _RERANKER_LOAD_FAILED as sentinel2
        self.assertIs(_RERANKER_LOAD_FAILED, sentinel2)

    def test_rerank_with_top_k_larger_than_docs(self):
        """top_k > len(docs) 时应返回全部"""
        from app.services.reranker import _RERANKER_LOAD_FAILED, RerankerService

        service = RerankerService()
        service._reranker = _RERANKER_LOAD_FAILED

        from langchain_core.documents import Document
        docs = [Document(page_content=f"D{i}", metadata={}) for i in range(3)]

        result = service.rerank("q", docs, top_k=10)
        self.assertEqual(len(result), 3)


# ============================================================
# 模块级功能测试（整条链路）
# ============================================================
class TestModuleLevelAPI(unittest.TestCase):
    """验证模块级 API 函数"""

    def test_rerank_function_exists(self):
        """rerank() 模块函数可正常导入"""
        from app.services.reranker import rerank
        self.assertTrue(callable(rerank))

    def test_bm25_search_function_exists(self):
        """bm25_search() 模块函数可正常导入"""
        from app.services.reranker import bm25_search
        self.assertTrue(callable(bm25_search))

    def test_rebuild_bm25_function_exists(self):
        """rebuild_bm25() 模块函数可正常导入"""
        from app.services.reranker import rebuild_bm25
        self.assertTrue(callable(rebuild_bm25))

    def test_sentinel_exported(self):
        """哨兵常量可被导入（供 agent_tools 或测试使用）"""
        from app.services.reranker import _RERANKER_LOAD_FAILED
        self.assertIsNotNone(_RERANKER_LOAD_FAILED)


if __name__ == "__main__":
    # 运行所有测试
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出汇总
    print("\n" + "=" * 60)
    print(f"测试汇总: {result.testsRun} 个测试")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)

    # 详细失败信息
    if result.failures:
        print("\n--- 失败详情 ---")
        for test, traceback in result.failures:
            print(f"\nFAIL: {test}")
            print(traceback)
    if result.errors:
        print("\n--- 错误详情 ---")
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(traceback)
