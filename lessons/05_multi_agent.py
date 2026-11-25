#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 5 课：多智能体 —— 拆角色，比堆一个大 Agent 更可靠。

    规划者 Planner  拆解任务
    执行者 Executor 逐条完成（可调工具）
    汇总者 Reviewer 组装交付

运行：
    python lessons/05_multi_agent.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helloagent import MockLLM, MultiAgent, Planner, config  # noqa: E402


def main() -> None:
    print("=" * 64)
    print("第 5 课 · 多智能体协作")
    print(config.describe())
    print("=" * 64)

    llm = MockLLM()

    # ---- 单独看规划者怎么拆 ----
    print("\n--- 规划者单独工作 ---")
    planner = Planner(llm=llm)
    for goal in ["规划一次景德镇两日游", "策划一场瓷乐音乐会", "我想入门 AI 智能体"]:
        plan = planner.plan(goal)
        print(f"\n目标：{goal}")
        for i, t in enumerate(plan.tasks, 1):
            print(f"  {i}. {t}")

    # ---- 三件套协作 ----
    print("\n--- 完整流水线 ---")
    team = MultiAgent(llm=llm, verbose=True)
    result = team.run("策划一场瓷乐音乐会，时长 90 分钟")
    print("\n最终交付：")
    print(result)

    print("-" * 64)
    print("观察与思考：")
    print("1. 规划者的拆解规则写死了会怎样？换成真实模型后应该改哪里？")
    print("2. 如果第 2 步执行失败，汇总者该如实报告还是掩盖？")
    print("3. 角色越多越好吗？多一个角色就多一次调用、多一份延迟与成本。")


if __name__ == "__main__":
    main()
