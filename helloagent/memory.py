# -*- coding: utf-8 -*-
"""
记忆：让智能体记住上下文。

三种记忆，按复杂度递进：
    NoMemory        无记忆，每次都是新的（最便宜的基线）
    WindowMemory    只保留最近 N 条消息（工程上最常用）
    SummaryMemory   超长时自动摘要，把"往事"压缩成一句话

注意：记忆是有成本的。历史越长，每次调用越贵、越慢，
所以真实项目不会无脑全量塞进去。
"""
from __future__ import annotations

from typing import List, Optional

from . import config
from .message import Conversation, Message


class BaseMemory:
    """记忆接口：决定每次调用模型时"带上哪些历史"。"""

    name = "base"

    def add_user(self, content: str) -> None:
        raise NotImplementedError

    def add_assistant(self, content: str, name: Optional[str] = None) -> None:
        raise NotImplementedError

    def build_messages(self) -> List[dict]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class NoMemory(BaseMemory):
    """零记忆：最省，也最容易"失忆"。"""

    name = "none"

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self._turn: List[Message] = []

    def add_user(self, content: str) -> None:
        self._turn = [Message("user", content)]

    def add_assistant(self, content: str, name: Optional[str] = None) -> None:
        self._turn.append(Message("assistant", content, name))

    def build_messages(self) -> List[dict]:
        out = [{"role": "system", "content": self.system_prompt}]
        out.extend(m.to_dict() for m in self._turn)
        return out

    def clear(self) -> None:
        self._turn = []


class WindowMemory(BaseMemory):
    """滑动窗口：只留最近 max_messages 条。"""

    name = "window"

    def __init__(self, max_messages: int = 0, system_prompt: str = "") -> None:
        self.limit = max_messages or config.MAX_HISTORY
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self.conv = Conversation(system_prompt=self.system_prompt)

    def add_user(self, content: str) -> None:
        self.conv.add_user(content)
        self._trim()

    def add_assistant(self, content: str, name: Optional[str] = None) -> None:
        self.conv.add_assistant(content, name)
        self._trim()

    def _trim(self) -> None:
        if len(self.conv.messages) > self.limit:
            # 从最老的开始丢，但不拆散
            drop = len(self.conv.messages) - self.limit
            self.conv.messages = self.conv.messages[drop:]

    def build_messages(self) -> List[dict]:
        return self.conv.to_api_messages()

    def clear(self) -> None:
        self.conv.messages.clear()


class SummaryMemory(BaseMemory):
    """摘要记忆：超出窗口时，把最早的一半压成一段摘要。

    真实项目里这一步通常也是一次 LLM 调用。
    这里为了离线可跑，用规则生成摘要。
    """

    name = "summary"

    def __init__(self, max_messages: int = 6, system_prompt: str = "") -> None:
        self.limit = max_messages
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self.history: List[Message] = []
        self.summary: str = ""

    def add_user(self, content: str) -> None:
        self.history.append(Message("user", content))
        self._maybe_compress()

    def add_assistant(self, content: str, name: Optional[str] = None) -> None:
        self.history.append(Message("assistant", content, name))
        self._maybe_compress()

    def _maybe_compress(self) -> None:
        if len(self.history) <= self.limit:
            return
        half = len(self.history) // 2
        old, keep = self.history[:half], self.history[half:]
        # 规则摘要：记录旧消息的轮次与主题
        topics = []
        for m in old:
            snippet = m.content.strip().replace("\n", " ")[:24]
            topics.append(f"{m.role}「{snippet}」")
        joined = "；".join(topics)
        self.summary = (
            f"{self.summary + ' ｜ ' if self.summary else ''}早先聊过：{joined}"
        )
        self.history = keep

    def build_messages(self) -> List[dict]:
        out = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            out.append(
                {
                    "role": "system",
                    "content": f"【更早对话的摘要】{self.summary}",
                }
            )
        out.extend(m.to_dict() for m in self.history)
        return out

    def clear(self) -> None:
        self.history.clear()
        self.summary = ""
