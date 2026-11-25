#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 4 课：记忆 —— 上下文为什么有成本。

对比三种记忆：
    NoMemory      每次从零开始
    WindowMemory  只留最近 N 条
    SummaryMemory 超长时压成摘要

运行：
    python lessons/04_memory.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helloagent import (  # noqa: E402
    MockLLM,
    NoMemory,
    SimpleAgent,
    SummaryMemory,
    WindowMemory,
    config,
)


def count_prompt_chars(memory) -> int:
    """粗略估算每次调用要发多少字符 —— 这就是成本。"""
    return sum(len(m["content"]) for m in memory.build_messages())


def run_dialogue(agent: SimpleAgent, label: str) -> None:
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    turns = [
        "我叫小林，在读音乐学。",
        "我在准备一场瓷乐音乐会。",
        "演出大概 90 分钟。",
        "现在算一下 90 分钟等于多少秒？",
    ]
    for t in turns:
        answer = agent.run(t)
        print(f">>> {t}")
        print(f"<<< {answer}")
        print(f"    [记忆：{len(agent.memory.build_messages())} 条 | 提示约 {count_prompt_chars(agent.memory)} 字符]")


def main() -> None:
    print("=" * 64)
    print("第 4 课 · 记忆管理")
    print(config.describe())
    print("=" * 64)

    # 1) 无记忆
    run_dialogue(
        SimpleAgent(llm=MockLLM(), memory=NoMemory(system_prompt="你是助手。")),
        "NoMemory（零记忆）",
    )

    # 2) 窗口记忆
    run_dialogue(
        SimpleAgent(llm=MockLLM(), memory=WindowMemory(max_messages=4, system_prompt="你是助手。")),
        "WindowMemory（保留最近 4 条）",
    )

    # 3) 摘要记忆
    mem = SummaryMemory(max_messages=2, system_prompt="你是助手。")
    agent = SimpleAgent(llm=MockLLM(), memory=mem)
    run_dialogue(agent, "SummaryMemory（超过 2 条就压缩）")
    print(f"\n压缩后的摘要：{mem.summary or '（还没触发）'}")

    print("\n" + "-" * 64)
    print("观察与思考：")
    print("1. 三种记忆，第 1 轮都要发多少字符？第 4 轮呢？")
    print("2. 窗口记忆把早期内容丢掉后，模型还能回答问题吗？")
    print("3. 真实产品里为什么常见『最近 N 轮 + 长期摘要 + 向量检索』三件套？")


if __name__ == "__main__":
    main()
