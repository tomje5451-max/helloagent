# -*- coding: utf-8 -*-
"""消息与对话历史的数据结构。"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    """一条对话消息。

    role 只有三种：
      - "system"    系统提示，设定人格与规则
      - "user"      用户说的话
      - "assistant" 模型/智能体的回答
    """

    role: str
    content: str
    name: Optional[str] = None  # 多智能体时标记是谁说的

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d

    def __str__(self) -> str:
        tag = f"[{self.role}:{self.name}]" if self.name else f"[{self.role}]"
        return f"{tag} {self.content}"


@dataclass
class Conversation:
    """对话历史。

    为什么单独做一个类？
    因为"记住上下文"是 Agent 和一次性问答的分水岭，
    把历史管理抽出来，后面每节课都能复用同一套。
    """

    system_prompt: str = ""
    messages: List[Message] = field(default_factory=list)

    def add(self, role: str, content: str, name: Optional[str] = None) -> Message:
        m = Message(role=role, content=content, name=name)
        self.messages.append(m)
        return m

    def add_user(self, content: str) -> Message:
        return self.add("user", content)

    def add_assistant(self, content: str, name: Optional[str] = None) -> Message:
        return self.add("assistant", content, name)

    def to_api_messages(self) -> List[dict]:
        """转成 OpenAI 接口要求的 messages 数组。"""
        out = []
        if self.system_prompt:
            out.append({"role": "system", "content": self.system_prompt})
        out.extend(m.to_dict() for m in self.messages)
        return out

    def transcript(self) -> str:
        """给模型看的纯文本历史（Mock 后端用）。"""
        lines = []
        for m in self.messages:
            who = m.name or m.role
            lines.append(f"{who}: {m.content}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.messages)
