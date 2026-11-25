# -*- coding: utf-8 -*-
"""
工具系统：让智能体"能做事"。

一个工具 = 名字 + 说明 + 参数格式 + 真正干活的函数。
Agent 只看得到"说明"和"参数格式"，看不到函数实现——
所以工具必须把用途写清楚，否则模型不知道什么时候该调用它。
"""
from __future__ import annotations

import ast
import datetime
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# --------------------------------------------------------------------------
# 工具定义
# --------------------------------------------------------------------------
@dataclass
class Tool:
    """一个可被调用的工具。

    name        : 唯一标识，模型靠它来"点名"工具
    description : 什么时候该用它，写给模型看
    parameters  : JSON Schema 风格的参数说明，也写给模型看
    func        : 真正执行的函数
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]

    def schema(self) -> Dict[str, Any]:
        """转成 OpenAI function calling 的工具描述。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """工具执行结果。ok=False 时 error 里是原因。"""

    ok: bool
    output: str
    error: str = ""

    def __str__(self) -> str:
        return self.output if self.ok else f"[工具出错] {self.error}"


# --------------------------------------------------------------------------
# 工具注册表
# --------------------------------------------------------------------------
class ToolRegistry:
    """工具注册表：把散落的工具收在一起，统一查找与调用。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"工具重名：{tool.name}")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def describe(self) -> str:
        """给模型看的工具清单（纯文本版）。"""
        lines = []
        for t in self._tools.values():
            params = ", ".join(t.parameters.get("properties", {}).keys()) or "无"
            lines.append(f"- {t.name}({params})：{t.description}")
        return "\n".join(lines)

    def call(self, name: str, args: Optional[Dict[str, Any]] = None) -> ToolResult:
        """调用工具。任何异常都在这里被收成 ToolResult，不往上炸。"""
        args = args or {}
        tool = self.get(name)
        if tool is None:
            return ToolResult(False, "", f"没有名为 {name} 的工具。可用：{', '.join(self.names())}")
        try:
            out = tool.func(**args)
            return ToolResult(True, str(out))
        except TypeError as e:
            return ToolResult(False, "", f"参数不匹配：{e}")
        except Exception as e:  # 工具内部错误不应该拖垮整个 Agent
            return ToolResult(False, "", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# 基础工具实现
# --------------------------------------------------------------------------
# 只允许这些运算，避免 eval 执行任意代码
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只支持数字")
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError("不支持的运算符")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError("不支持的一元运算符")
        return op(_safe_eval(node.operand))
    raise ValueError("表达式过于复杂")


def calculator(expression: str) -> str:
    """安全地计算四则运算表达式。

    容错：模型经常漏写右括号（如 "12*(3+4"），这里自动补齐，
    而不是直接把错误抛给用户。
    """
    expr = expression.strip()
    # 补齐未闭合的括号
    balance = expr.count("(") - expr.count(")")
    if balance > 0:
        expr += ")" * balance
    tree = ast.parse(expr, mode="eval")
    value = _safe_eval(tree)
    # 整数就不显示小数尾巴
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def get_weather(city: str) -> str:
    """Mock 天气数据。真实场景应调用天气 API。"""
    data = {
        "景德镇": ("多云", 18, 26, "东风 2 级"),
        "北京": ("晴", 5, 16, "西北风 3 级"),
        "上海": ("小雨", 14, 19, "东南风 2 级"),
        "深圳": ("多云", 22, 28, "微风"),
    }
    if city not in data:
        return f"暂无 {city} 的天气数据（这是 Mock 工具，只内置了 {len(data)} 个城市）"
    sky, low, high, wind = data[city]
    return f"{city}：{sky}，{low}~{high}℃，{wind}"


def get_time() -> str:
    """返回当前日期与时间。"""
    now = datetime.datetime.now()
    week = "一二三四五六日"[now.weekday()]
    return now.strftime(f"%Y-%m-%d %H:%M:%S（星期{week}）")


def build_default_registry() -> ToolRegistry:
    """默认注册三个工具，够前几课用了。"""
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="calculator",
            description="计算四则运算。当用户需要算数、核对金额、换算单位时使用。只支持数字与 + - * / ( ) ** %",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "算式，例如 12*(3+4)"}
                },
                "required": ["expression"],
            },
            func=calculator,
        )
    )
    reg.register(
        Tool(
            name="get_weather",
            description="查询指定城市的天气。当用户问天气、气温、要不要带伞时使用。",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，例如 景德镇"}
                },
                "required": ["city"],
            },
            func=get_weather,
        )
    )
    reg.register(
        Tool(
            name="get_time",
            description="获取当前的日期与时间。当用户问今天几号、现在几点时使用。",
            parameters={"type": "object", "properties": {}, "required": []},
            func=get_time,
        )
    )
    return reg
