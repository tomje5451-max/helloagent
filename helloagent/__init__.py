# -*- coding: utf-8 -*-
"""
helloagent —— 从零理解智能体（Agent）的最小实现。

四个核心概念，对应四个模块：

    message.py  消息与对话历史 —— Agent 的"记性"
    tools.py    工具系统       —— Agent 的"手"
    llm.py      大模型客户端   —— Agent 的"脑"
    agent.py    最小智能体     —— 把上面三样装起来
    react.py    推理循环       —— 会"边想边做"的 Agent
    multi.py    多智能体       —— 会"分工协作"的 Agent

快速开始：

    from helloagent import SimpleAgent
    agent = SimpleAgent()
    print(agent.run("帮我算一下 24*7"))
"""
from .agent import SimpleAgent
from .llm import BaseLLM, MockLLM, OpenAILLM, Reply, build_llm
from .memory import BaseMemory, NoMemory, SummaryMemory, WindowMemory
from .message import Conversation, Message
from .multi import Executor, MultiAgent, Planner, Reviewer
from .react import ReActAgent, Step
from .tools import Tool, ToolRegistry, ToolResult, build_default_registry

__version__ = "0.1.0"

__all__ = [
    "SimpleAgent",
    "ReActAgent",
    "MultiAgent",
    "Planner",
    "Executor",
    "Reviewer",
    "BaseLLM",
    "MockLLM",
    "OpenAILLM",
    "Reply",
    "build_llm",
    "BaseMemory",
    "NoMemory",
    "WindowMemory",
    "SummaryMemory",
    "Message",
    "Conversation",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "Step",
]
