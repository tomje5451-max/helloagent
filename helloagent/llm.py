# -*- coding: utf-8 -*-
"""
大模型客户端：一个接口，两种实现。

    llm = MockLLM()      # 离线，规则模拟，零成本
    llm = OpenAILLM()    # 接真实模型，需要密钥

两者都只需要实现 chat() 方法，Agent 不关心背后是哪个。
这就是"面向接口编程"——换模型不改 Agent 代码。
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import config


@dataclass
class Reply:
    """模型的一次回复。"""

    content: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content


# --------------------------------------------------------------------------
# 接口
# --------------------------------------------------------------------------
class BaseLLM:
    """所有 LLM 实现的基类。"""

    name = "base"

    def chat(self, messages: List[dict]) -> Reply:
        raise NotImplementedError


# --------------------------------------------------------------------------
# 离线 Mock：用规则模拟大模型
# --------------------------------------------------------------------------
class MockLLM(BaseLLM):
    """规则模拟的大模型，让整套实验无需密钥就能跑通。

    它当然不"智能"，但它遵守和真模型一样的**输入输出契约**：
    输入 messages，输出一段文本。所以：
      - 你现在学的调用方式是真实可用的；
      - 把 MockLLM 换成 OpenAILLM，Agent 代码一行不改。

    匹配规则按顺序尝试，先命中的生效。
    """

    name = "mock"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.calls = 0

    # -- 对外接口 --
    def chat(self, messages: List[dict]) -> Reply:
        self.calls += 1
        last = self._last_user_text(messages)
        history = self._history_text(messages)

        for rule_name, pattern, handler in self._rules():
            if re.search(pattern, last):
                content = handler(last, history)
                if self.verbose:
                    print(f"[mock] 命中规则：{rule_name}")
                return Reply(content, meta={"rule": rule_name, "model": self.name})

        return Reply(
            "这个问题我需要更多信息才能回答。你可以告诉我具体想了解什么吗？",
            meta={"rule": "fallback", "model": self.name},
        )

    # -- 工具方法 --
    @staticmethod
    def _last_user_text(messages: List[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    @staticmethod
    def _history_text(messages: List[dict]) -> str:
        return "\n".join(
            f"{m.get('role')}: {m.get('content')}" for m in messages
        )

    # -- 规则表 --
    def _rules(self):
        """(规则名, 正则, 处理函数)，按优先级排列。"""
        return [
            ("react", r"Thought|请先给出一句|推理轨迹|scratcpad|scratchpad", self._react_rule),
            ("tool_name", r"有哪些工具|你会什么工具|工具列表|能调用什么", self._tool_list_rule),
            ("math", r"算|计算|多少|等于|乘以|乘以多少|[0-9]\s*[\*x×+/-]\s*[0-9]", self._math_rule),
            ("weather", r"天气|气温|下雨", self._weather_rule),
            ("time", r"几点|现在时间|今天几号|日期", self._time_rule),
            ("self_intro", r"你是谁|你叫什么|介绍一下你|helloagent 是什么|hello", self._intro_rule),
            ("summary", r"总结|概括|回顾一下", self._summary_rule),
        ]

    # -- 各规则的处理逻辑 --
    def _react_rule(self, last: str, history: str) -> str:
        # 已经拿到工具返回的观察结果 → 该收尾了
        if "Observation:" in history:
            m = re.search(r"Observation:\s*(.+)", history)
            observation = m.group(1).strip() if m else ""
            if observation and not observation.startswith("错误"):
                return f"Final Answer: {observation}"
            return "Final Answer: 工具没有返回有效结果，建议检查参数后重试。"

        expr = self._extract_expr(last)
        if expr:
            return (
                "Thought: 这是一个计算问题，我需要调用计算器。\n"
                "Action: calculator\n"
                f'Action Input: {{"expression": "{expr}"}}'
            )
        if re.search(r"天气|气温", last):
            return (
                "Thought: 需要查询天气工具。\n"
                "Action: get_weather\n"
                'Action Input: {"city": "景德镇"}'
            )
        if re.search(r"几点|时间|日期", last):
            return "Thought: 需要获取当前时间。\nAction: get_time\nAction Input: {}"
        return "Final Answer: 我还不会处理这个问题，换个问题试试？"

    def _tool_list_rule(self, last: str, history: str) -> str:
        return json.dumps(
            {
                "可用工具": [
                    {"name": "calculator", "desc": "计算四则运算表达式"},
                    {"name": "get_weather", "desc": "查询城市天气（Mock 数据）"},
                    {"name": "get_time", "desc": "获取当前时间"},
                ]
            },
            ensure_ascii=False,
        )

    def _math_rule(self, last: str, history: str) -> str:
        expr = self._extract_expr(last)
        if expr:
            return f"用计算器算一下：\n```json\n{{\"tool\": \"calculator\", \"input\": {{\"expression\": \"{expr}\"}}}}\n```"
        return "你想计算什么？请把算式告诉我。"

    def _weather_rule(self, last: str, history: str) -> str:
        return (
            "```json\n"
            '{"tool": "get_weather", "input": {"city": "景德镇"}}\n'
            "```"
        )

    def _time_rule(self, last: str, history: str) -> str:
        return "```json\n{\"tool\": \"get_time\", \"input\": {}}\n```"

    def _intro_rule(self, last: str, history: str) -> str:
        return (
            "我是 helloagent —— 一个用来学习「智能体」的最小实现。"
            "它由四部分组成：消息、工具、推理循环、记忆。"
            "你可以读源码、改规则、再接上真实大模型。"
        )

    def _summary_rule(self, last: str, history: str) -> str:
        turns = history.count("user:")
        return (
            f"这次对话共 {turns} 轮。"
            "核心结论：Agent = 大模型 + 工具 + 记忆 + 循环控制。"
            "缺任何一项，它就只是一个聊天机器人。"
        )

    @staticmethod
    def _extract_expr(text: str) -> Optional[str]:
        """从中文句子里抠出算式，例如“算一下 12*(3+4)”。"""
        normalized = text.replace("×", "*").replace("x", "*").replace("（", "(").replace("）", ")")
        normalized = re.sub(r"\s+", "", normalized)
        # 允许数字、运算符与括号，且必须以数字开头
        m = re.search(r"(\d[\d\.\+\-\*/\(\)%]*)", normalized)
        if not m:
            return None
        expr = m.group(1).strip("+-*/%")
        return expr or None


# --------------------------------------------------------------------------
# 真实模型：OpenAI 兼容接口
# --------------------------------------------------------------------------
class OpenAILLM(BaseLLM):
    """调用任何 OpenAI 兼容的 /chat/completions 接口。

    只需三个环境变量：
        HELLOAGENT_API_KEY
        HELLOAGENT_BASE_URL   （默认 OpenAI 官方）
        HELLOAGENT_MODEL
    """

    name = "openai"

    def __init__(self, api_key: str = "", base_url: str = "", model: str = "", timeout: int = 60):
        self.api_key = api_key or config.API_KEY
        self.base_url = (base_url or config.BASE_URL).rstrip("/")
        self.model = model or config.MODEL
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                "缺少 API Key。请设置环境变量 HELLOAGENT_API_KEY，"
                "或改用 MockLLM() 离线实验。"
            )

    def chat(self, messages: List[dict]) -> Reply:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            {"model": self.model, "messages": messages, "temperature": 0.2},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - 网络分支
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"模型接口返回 {e.code}：{detail}") from e

        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        return Reply(content, meta={"model": self.model, "usage": usage})


def build_llm() -> BaseLLM:
    """按配置返回一个 LLM。"""
    if config.BACKEND == "openai":
        return OpenAILLM()
    return MockLLM()
