# -*- coding: utf-8 -*-
"""
ReAct 智能体：边想边做，做完再看结果继续想。

ReAct = Reasoning + Acting，核心是一个**循环**：

    Thought: 我需要先算一下
    Action:  calculator
    Action Input: {"expression": "24*7"}
    Observation: 168          ← 这是工具返回的，由程序填回去
    Thought: 算出来了，可以回答了
    Final Answer: 24*7 = 168

和 SimpleAgent 的区别：
    SimpleAgent 最多调一次工具；
    ReActAgent 可以"调工具 → 看结果 → 再调工具"，直到拿到答案。
    代价是可能绕圈，所以必须有 max_steps 兜底。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from . import config
from .llm import BaseLLM, MockLLM
from .tools import ToolRegistry, build_default_registry

# 解析模型输出里的三段式指令
_RE_THOUGHT = re.compile(r"Thought:\s*(.+)")
_RE_ACTION = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)")
_RE_INPUT = re.compile(r"Action Input:\s*(\{.*?\})", re.S)
_RE_FINAL = re.compile(r"Final Answer:\s*(.+)", re.S)


@dataclass
class Step:
    """ReAct 循环中的一步，便于调试与展示。"""

    index: int
    thought: str = ""
    action: str = ""
    action_input: dict = field(default_factory=dict)
    observation: str = ""
    final_answer: str = ""
    error: str = ""

    def summary(self) -> str:
        if self.final_answer:
            return f"第{self.index}步 · 得出答案：{self.final_answer[:40]}"
        if self.action:
            return f"第{self.index}步 · 调用 {self.action}({self.action_input}) → {self.observation[:40]}"
        return f"第{self.index}步 · {self.thought[:40]}"


class ReActAgent:
    """带推理循环的智能体。"""

    def __init__(
        self,
        name: str = "react",
        llm: Optional[BaseLLM] = None,
        registry: Optional[ToolRegistry] = None,
        max_steps: int = 0,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.llm = llm or MockLLM()
        self.registry = registry or build_default_registry()
        self.max_steps = max_steps or config.MAX_STEPS
        self.verbose = verbose
        self.steps: List[Step] = []  # 保留轨迹，方便复盘

    # ------------------------------------------------------------------
    def run(self, question: str) -> str:
        """执行一次完整的 ReAct 循环。"""
        self.steps = []
        scratchpad = ""  # 累积的思考轨迹，每轮都喂回模型

        for i in range(1, self.max_steps + 1):
            step = Step(index=i)
            prompt = self._build_prompt(question, scratchpad)
            reply = self.llm.chat(prompt)
            text = reply.content
            if self.verbose:
                print(f"\n--- 第 {i} 步模型输出 ---\n{text}")

            # 1) 先看有没有给出最终答案
            m_final = _RE_FINAL.search(text)
            if m_final:
                answer = m_final.group(1).strip()
                # 如果同一段里既有 Action 又有 Final Answer，优先 Final
                step.final_answer = answer
                self.steps.append(step)
                return answer

            # 2) 解析 Thought / Action / Action Input
            m_thought = _RE_THOUGHT.search(text)
            m_action = _RE_ACTION.search(text)
            if not m_action:
                # 既没答案也没动作：把模型这段话当作答案，但记录一次空转
                step.error = "模型既没有给出 Action，也没有给出 Final Answer"
                self.steps.append(step)
                return text.strip() or "（模型没有输出）"

            tool_name = m_action.group(1)
            step.thought = (m_thought.group(1).strip() if m_thought else "")
            step.action = tool_name

            m_input = _RE_INPUT.search(text)
            args: dict = {}
            if m_input:
                try:
                    args = json.loads(m_input.group(1))
                except json.JSONDecodeError:
                    args = {}
            step.action_input = args

            # 3) 执行工具，把结果写回轨迹
            result = self.registry.call(tool_name, args)
            step.observation = result.output if result.ok else f"错误：{result.error}"
            self.steps.append(step)
            if self.verbose:
                print(f"[{self.name}] {step.summary()}")

            scratchpad += (
                f"\nThought: {step.thought}\n"
                f"Action: {tool_name}\n"
                f"Action Input: {json.dumps(args, ensure_ascii=False)}\n"
                f"Observation: {step.observation}\n"
            )

        # 用完步数还没答案：明确告诉用户，而不是假装成功
        return (
            f"我在 {self.max_steps} 步内没有得出确定答案。"
            "已记录每一步的轨迹，可以调大 max_steps 或检查工具是否够用。"
        )

    # ------------------------------------------------------------------
    def _build_prompt(self, question: str, scratchpad: str) -> List[dict]:
        """构造带工具说明与历史轨迹的提示。"""
        system = (
            f"{config.SYSTEM_PROMPT}\n\n"
            "你是一个会使用工具的智能体。请严格按下面的格式思考：\n"
            "Thought: 你的思考\n"
            "Action: 工具名（必须是下面列表中的一个）\n"
            'Action Input: {"参数": "值"}\n\n'
            "当你有足够信息回答时，输出：\n"
            "Final Answer: 你的最终回答\n\n"
            f"可用工具：\n{self.registry.describe()}\n"
            "注意：一次只做一个 Action，不要自己编造 Observation。"
        )
        user = f"问题：{question}\n"
        if scratchpad:
            user += f"\n已知的推理轨迹：{scratchpad}\n请继续。"
        else:
            user += "\n请先给出一句 Thought。"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ------------------------------------------------------------------
    def trace(self) -> str:
        """把推理轨迹渲染成可读文本，写作业/复盘很好用。"""
        if not self.steps:
            return "（没有轨迹）"
        lines = []
        for s in self.steps:
            lines.append(f"【第 {s.index} 步】")
            if s.thought:
                lines.append(f"  想法：{s.thought}")
            if s.action:
                lines.append(f"  动作：{s.action}  参数：{s.action_input}")
            if s.observation:
                lines.append(f"  观察：{s.observation}")
            if s.error:
                lines.append(f"  异常：{s.error}")
            if s.final_answer:
                lines.append(f"  结论：{s.final_answer}")
        return "\n".join(lines)
