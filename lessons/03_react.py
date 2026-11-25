#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 3 课：ReAct —— 让智能体学会"边想边做"。

本课重点看**轨迹（trace）**：每一步它在想什么、调用了什么、看到了什么。
这就是 Agent 可调试、可复盘的来源。

运行：
    python lessons/03_react.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helloagent import MockLLM, ReActAgent, config  # noqa: E402


def main() -> None:
    print("=" * 64)
    print("第 3 课 · ReAct 推理循环")
    print(config.describe())
    print("=" * 64)

    agent = ReActAgent(llm=MockLLM(), max_steps=4, verbose=True)

    print("\n--- 问题 1：需要调用计算器 ---")
    q = "算一下 128*4 是多少"
    print(f">>> {q}")
    answer = agent.run(q)
    print(f"<<< {answer}")
    print("\n推理轨迹：")
    print(agent.trace())

    print("\n--- 问题 2：需要查天气 ---")
    q = "景德镇今天天气如何"
    print(f">>> {q}")
    print(f"<<< {agent.run(q)}")
    print("\n推理轨迹：")
    print(agent.trace())

    print("\n--- 问题 3：模型不会处理的任务（看它如何收场）---")
    q = "帮我写一首关于瓷器的诗"
    print(f">>> {q}")
    print(f"<<< {agent.run(q)}")

    print("\n--- 问题 4：把 max_steps 调到 1，强制它中途放弃 ---")
    tiny = ReActAgent(llm=MockLLM(), max_steps=1)
    print(f">>> {q}")
    print(f"<<< {tiny.run(q)}")
    print("注意：步数不够时它明确说『没得出答案』，而不是编一个。")

    print("\n" + "-" * 64)
    print("观察与思考：")
    print("1. 轨迹里的 Observation 是谁填的？模型还是程序？")
    print("2. 如果没有 max_steps，循环在什么条件下才会停？")
    print("3. 把工具调用改成两次（先算数再查天气），循环会怎么走？")


if __name__ == "__main__":
    main()
