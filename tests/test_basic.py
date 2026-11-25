# -*- coding: utf-8 -*-
"""
单元测试：保证改动不会悄悄弄坏已有能力。

全离线、零依赖，直接跑：
    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helloagent import (  # noqa: E402
    MockLLM,
    MultiAgent,
    NoMemory,
    ReActAgent,
    SimpleAgent,
    SummaryMemory,
    Tool,
    ToolRegistry,
    WindowMemory,
    build_default_registry,
)


class TestTools(unittest.TestCase):
    def test_calculator(self):
        reg = build_default_registry()
        self.assertEqual(str(reg.call("calculator", {"expression": "24*7"})), "168")
        self.assertEqual(str(reg.call("calculator", {"expression": "(3+4)*2"})), "14")

    def test_unknown_tool(self):
        reg = build_default_registry()
        r = reg.call("nope", {})
        self.assertFalse(r.ok)
        self.assertIn("没有名为", r.error)

    def test_wrong_params(self):
        reg = build_default_registry()
        r = reg.call("calculator", {"bad": "1"})
        self.assertFalse(r.ok)

    def test_dangerous_expression_rejected(self):
        reg = build_default_registry()
        r = reg.call("calculator", {"expression": "__import__('os').system('echo hi')"})
        self.assertFalse(r.ok, "非数字表达式必须被拒绝")

    def test_duplicate_tool_name(self):
        reg = ToolRegistry()
        t = Tool("x", "d", {"type": "object", "properties": {}}, lambda: "1")
        reg.register(t)
        with self.assertRaises(ValueError):
            reg.register(t)

    def test_weather_known_and_unknown(self):
        reg = build_default_registry()
        self.assertIn("景德镇", reg.call("get_weather", {"city": "景德镇"}).output)
        self.assertIn("暂无", reg.call("get_weather", {"city": "火星"}).output)


class TestSimpleAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleAgent(llm=MockLLM(), memory=WindowMemory(max_messages=6))

    def test_math_uses_tool(self):
        answer = self.agent.run("帮我算一下 24*7")
        self.assertIn("168", answer)

    def test_weather(self):
        answer = self.agent.run("景德镇天气怎么样")
        self.assertIn("景德镇", answer)

    def test_plain_question_no_tool(self):
        answer = self.agent.run("你是谁")
        self.assertIn("helloagent", answer)

    def test_memory_keeps_history(self):
        self.agent.run("第一句")
        self.agent.run("第二句")
        self.assertGreaterEqual(len(self.agent.memory.build_messages()), 3)

    def test_no_memory_forgets(self):
        agent = SimpleAgent(llm=MockLLM(), memory=NoMemory())
        agent.run("第一句")
        agent.run("第二句")
        # system + 当前一轮（user + assistant）
        self.assertLessEqual(len(agent.memory.build_messages()), 3)


class TestReAct(unittest.TestCase):
    def test_math_loop(self):
        agent = ReActAgent(llm=MockLLM(), max_steps=4)
        answer = agent.run("算一下 128*4")
        self.assertIn("128", agent.trace())
        self.assertTrue(answer)

    def test_trace_recorded(self):
        agent = ReActAgent(llm=MockLLM(), max_steps=3)
        agent.run("算一下 2*3")
        self.assertGreaterEqual(len(agent.steps), 1)
        self.assertTrue(any(s.action for s in agent.steps))

    def test_respects_max_steps(self):
        # 第 1 步只给 Action，模型不会自己给 Final，步数用完必须收手
        class AlwaysAct(MockLLM):
            def chat(self, messages):
                from helloagent.llm import Reply

                return Reply("Thought: 继续查\nAction: get_time\nAction Input: {}")

        agent = ReActAgent(llm=AlwaysAct(), max_steps=2)
        answer = agent.run("随便")
        self.assertIn("没有得出确定答案", answer)
        self.assertEqual(len(agent.steps), 2)


class TestMemory(unittest.TestCase):
    def test_window_trim(self):
        mem = WindowMemory(max_messages=2)
        for i in range(5):
            mem.add_user(f"u{i}")
            mem.add_assistant(f"a{i}")
        self.assertEqual(len(mem.conv.messages), 2)

    def test_summary_compresses(self):
        mem = SummaryMemory(max_messages=2)
        for i in range(6):
            mem.add_user(f"话题{i}")
            mem.add_assistant(f"回答{i}")
        self.assertTrue(mem.summary)
        self.assertLessEqual(len(mem.history), 2)
        # 摘要要体现在发给模型的 messages 里
        blob = " ".join(m["content"] for m in mem.build_messages())
        self.assertIn("摘要", blob)


class TestMultiAgent(unittest.TestCase):
    def test_pipeline_produces_result(self):
        team = MultiAgent(llm=MockLLM())
        out = team.run("策划一场瓷乐音乐会")
        self.assertIn("瓷乐音乐会", out)
        self.assertIn("子任务", out)
        self.assertIsNotNone(team.last_plan)

    def test_plan_has_multiple_tasks(self):
        from helloagent.multi import Planner

        plan = Planner(llm=MockLLM()).plan("规划一次景德镇两日游")
        self.assertGreaterEqual(len(plan.tasks), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
