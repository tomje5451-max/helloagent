#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 2 课：工具调用 —— 为什么"工具"必须写清楚用途。

同一句话，工具的 description 写得不同，模型选错工具的概率也不同。
本课通过对比实验让你看到：描述就是提示词。

运行：
    python lessons/02_tools.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helloagent import MockLLM, SimpleAgent, Tool, ToolRegistry, config  # noqa: E402


def uppercase(text: str) -> str:
    return text.upper()


def word_count(text: str) -> str:
    return str(len(text.split()))


def main() -> None:
    print("=" * 64)
    print("第 2 课 · 工具系统")
    print(config.describe())
    print("=" * 64)

    # ---- 实验 A：注册两个自定义工具，看模型能不能选对 ----
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="to_upper",
            description="把一段英文转成大写。当用户要求『转大写 / shout it』时使用。",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "要转换的文本"}},
                "required": ["text"],
            },
            func=uppercase,
        )
    )
    reg.register(
        Tool(
            name="count_words",
            description="统计一段文字里有多少个单词。当用户问『多少词 / 字数』时使用。",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "要统计的文本"}},
                "required": ["text"],
            },
            func=word_count,
        )
    )

    print("\n--- 实验 A：注册的自定义工具 ---")
    print(reg.describe())

    print("\n--- 直接调用（不经过模型）---")
    print("to_upper('hello agent') ->", reg.call("to_upper", {"text": "hello agent"}))
    print("count_words('hello agent world') ->", reg.call("count_words", {"text": "hello agent world"}))

    print("\n--- 实验 B：故意调用不存在的工具 ---")
    bad = reg.call("send_email", {"to": "someone"})
    print("结果：", bad)
    print("注意：工具不存在时返回 ok=False，而不是抛异常 —— 智能体不应该因为一次失败就崩。")

    print("\n--- 实验 C：参数给错 ---")
    wrong = reg.call("to_upper", {"wrong_param": "abc"})
    print("结果：", wrong)

    # ---- 实验 D：工具清单会进入系统提示 ----
    print("\n--- 实验 D：系统提示里长什么样 ---")
    agent = SimpleAgent(registry=reg, llm=MockLLM())
    prompt = agent.system_prompt()
    print(prompt[:400] + "...")
    print("\n结论：模型的『能力边界』，是由你给的工具清单决定的。")

    print("\n" + "-" * 64)
    print("观察与思考：")
    print("1. 把 to_upper 的 description 改成『处理文本』，模型还选得对吗？")
    print("2. 两个工具描述高度重叠会怎样？（提示：模型只能猜）")
    print("3. 为什么工具内部异常要被捕获成 ToolResult，而不是直接抛出？")


if __name__ == "__main__":
    main()
