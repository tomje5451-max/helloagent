#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 1 课：最小可用 —— 一个能说话、会算数的智能体。

要观察的三件事：
  1. 智能体本质就是「把历史 + 工具说明拼成提示，交给模型」
  2. 模型输出可以是一段自然语言，也可以是一个工具调用指令
  3. 工具调用失败不应该让程序崩掉

运行：
    python lessons/01_minimal.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helloagent import MockLLM, SimpleAgent, WindowMemory, config  # noqa: E402


def main() -> None:
    print("=" * 64)
    print("第 1 课 · 最小智能体")
    print(config.describe())
    print("=" * 64)

    agent = SimpleAgent(
        name="demo",
        llm=MockLLM(verbose=True),
        memory=WindowMemory(max_messages=6),
        verbose=True,
    )

    questions = [
        "你好，你是谁？",
        "帮我算一下 24*7 是多少？",
        "复杂一点：12*(3+4) 等于多少",
        "景德镇今天天气怎么样？",
        "现在几点了？",
    ]

    for q in questions:
        print(f"\n>>> 用户：{q}")
        answer = agent.run(q)
        print(f"<<< 智能体：{answer}")

    print("\n" + "-" * 64)
    print("观察与思考：")
    print("1. 哪些问题触发了工具？哪些没有？判断依据是什么？")
    print("2. 如果模型输出格式写错（比如不是合法 JSON），会怎样？")
    print("3. 把 memory 换成 NoMemory()，再问一次『刚才我问了什么』，看看区别。")


if __name__ == "__main__":
    main()
