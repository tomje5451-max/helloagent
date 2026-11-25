# -*- coding: utf-8 -*-
"""
多智能体协作：分工，再汇总。

一个 Agent 什么都干，往往什么都干不好。
更常见也更可靠的思路是**拆角色**：

    规划者 Planner   → 把任务拆成子任务清单
    执行者 Executor  → 逐个完成子任务
    汇总者 Reviewer  → 检查并组织成最终结果

这不是"智能"的魔法，本质是**把大问题拆小、让每步的提示更聚焦**。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .agent import SimpleAgent
from .llm import BaseLLM, MockLLM
from .tools import ToolRegistry, build_default_registry


@dataclass
class Plan:
    """一份任务计划。"""

    goal: str
    tasks: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"目标：{self.goal}"]
        for i, t in enumerate(self.tasks, 1):
            lines.append(f"  {i}. {t}")
        return "\n".join(lines)


class Planner:
    """把一句话目标拆成 2~4 个可执行子任务。"""

    def __init__(self, llm: Optional[BaseLLM] = None) -> None:
        self.llm = llm or MockLLM()

    def plan(self, goal: str) -> Plan:
        # 真实项目靠模型拆解；这里用规则 + 模型兜底，保证离线可跑
        rules = [
            (r"旅行|旅游|出行|攻略", ["确定目的地与时间", "查询当地天气", "估算预算", "给出建议"]),
            (r"音乐会|演出|活动|策划", ["明确主题与受众", "安排节目与人员", "估算预算", "列出风险"]),
            (r"学习|入门|怎么学|路线", ["明确目标", "拆解知识点", "安排练习", "给出检验方式"]),
        ]
        for pattern, tasks in rules:
            if re.search(pattern, goal):
                return Plan(goal=goal, tasks=list(tasks))
        # 兜底：通用四步
        return Plan(
            goal=goal,
            tasks=[f"理解目标：{goal}", "收集需要的信息", "给出方案", "检查是否可执行"],
        )


class Executor:
    """逐条执行子任务；需要算数/查天气时自动走工具。"""

    def __init__(self, llm: Optional[BaseLLM] = None, registry: Optional[ToolRegistry] = None):
        self.agent = SimpleAgent(name="executor", llm=llm, registry=registry)
        # 多智能体里每个角色独立记忆，避免互相污染
        from .memory import WindowMemory

        self.agent.memory = WindowMemory(max_messages=4)

    def execute(self, plan: Plan) -> List[dict]:
        results = []
        for task in plan.tasks:
            answer = self.agent.run(task)
            results.append({"task": task, "result": answer})
        return results


class Reviewer:
    """把执行结果组织成最终交付。"""

    def __init__(self, llm: Optional[BaseLLM] = None) -> None:
        self.llm = llm or MockLLM()

    def review(self, plan: Plan, results: List[dict]) -> str:
        lines = [f"【{plan.goal}】", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['task']}")
            lines.append(f"   → {r['result']}")
        lines.append("")
        lines.append(f"共完成 {len(results)} 项子任务。哪些还需要你亲自确认，请在提交前核对。")
        return "\n".join(lines)


class MultiAgent:
    """三件套：规划 → 执行 → 汇总。"""

    def __init__(self, llm: Optional[BaseLLM] = None, verbose: bool = False) -> None:
        llm = llm or MockLLM()
        registry = build_default_registry()
        self.planner = Planner(llm=llm)
        self.executor = Executor(llm=llm, registry=registry)
        self.reviewer = Reviewer(llm=llm)
        self.verbose = verbose
        self.last_plan: Optional[Plan] = None

    def run(self, goal: str) -> str:
        plan = self.planner.plan(goal)
        self.last_plan = plan
        if self.verbose:
            print(f"\n[planner] 拆解为 {len(plan.tasks)} 个子任务：\n{plan}")
        results = self.executor.execute(plan)
        if self.verbose:
            for r in results:
                print(f"[executor] {r['task']} -> {r['result'][:60]}")
        return self.reviewer.review(plan, results)
