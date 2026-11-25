# -*- coding: utf-8 -*-
"""
最简单的智能体：一次对话 + 可选的一次工具调用。

它解决的是最基础的问题：
    用户说的话 → 模型 → 要么直接回答，要么调用工具 → 回答

虽然简单，但它已经包含了 Agent 的三个要素：
    1. 模型（llm）
    2. 工具（registry）
    3. 状态（memory）
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from . import config
from .llm import BaseLLM, MockLLM
from .memory import BaseMemory, WindowMemory
from .message import Message
from .tools import ToolRegistry, ToolResult, build_default_registry

# 匹配 ```json ... ``` 代码块里的工具调用
_TOOL_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


class SimpleAgent:
    """单轮工具调用智能体。

    用法：
        agent = SimpleAgent()
        print(agent.run("帮我算一下 24*7"))
    """

    def __init__(
        self,
        name: str = "simple",
        llm: Optional[BaseLLM] = None,
        registry: Optional[ToolRegistry] = None,
        memory: Optional[BaseMemory] = None,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.llm = llm or MockLLM()
        self.registry = registry or build_default_registry()
        self.memory = memory or WindowMemory()
        self.verbose = verbose
        self.last_tool_result: Optional[ToolResult] = None

    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        """把工具清单塞进系统提示，模型才知道有什么可用。"""
        return (
            f"{config.SYSTEM_PROMPT}\n\n"
            f"你可以调用以下工具（每次最多一个）：\n{self.registry.describe()}\n\n"
            "需要调用工具时，只输出一个 JSON 代码块，格式：\n"
            '```json\n{"tool": "工具名", "input": {"参数": "值"}}\n```\n'
            "不需要工具时，直接用自然语言回答。"
        )

    # ------------------------------------------------------------------
    def run(self, user_input: str) -> str:
        """处理一次用户输入，返回最终回答。"""
        self.memory.add_user(user_input)

        messages = [{"role": "system", "content": self.system_prompt()}]
        messages.extend(self.memory.build_messages()[1:])  # 去掉 memory 自带的 system

        reply = self.llm.chat(messages)
        if self.verbose:
            print(f"[{self.name}] 模型原始输出：{reply.content}")

        call = self._parse_tool_call(reply.content)
        if call is None:
            self.memory.add_assistant(reply.content)
            return reply.content

        # 有工具调用：先执行，再让模型总结
        tool_name, args = call
        result = self.registry.call(tool_name, args)
        self.last_tool_result = result
        if self.verbose:
            print(f"[{self.name}] 调用工具 {tool_name}({args}) -> {result}")

        final = self._summarize(tool_name, args, result, user_input)
        self.memory.add_assistant(final)
        return final

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_tool_call(text: str) -> Optional[Tuple[str, dict]]:
        """从模型输出里解析出 (工具名, 参数)。"""
        m = _TOOL_BLOCK.search(text)
        if not m:
            return None
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
        tool = data.get("tool")
        if not isinstance(tool, str):
            return None
        args = data.get("input") or {}
        if not isinstance(args, dict):
            args = {}
        return tool, args

    # ------------------------------------------------------------------
    def _summarize(self, tool_name: str, args: dict, result: ToolResult, question: str) -> str:
        """工具结果 → 自然语言回答。

        Mock 模式下用规则拼；真实模型下应该把结果再喂回模型润色。
        这里刻意用一个独立方法，方便第 5 课替换成"二次调用模型"。
        """
        if not result.ok:
            return f"我尝试调用 {tool_name} 但失败了：{result.error}。要不换个方式问？"

        if tool_name == "calculator":
            expr = args.get("expression", "")
            return f"{expr} = {result.output}"
        if tool_name == "get_weather":
            return f"查到了天气：{result.output}"
        if tool_name == "get_time":
            return f"现在是 {result.output}"

        # 未知工具：原样返回
        return result.output
