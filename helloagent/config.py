# -*- coding: utf-8 -*-
"""
全局配置。

设计原则：默认零依赖、零密钥就能跑通全部实验。
只有在你想接入真实大模型时，才需要设置环境变量。
"""
import os

# ---- 运行模式 ----
# "mock"      : 用内置规则模拟大模型，无需网络与密钥（默认）
# "openai"    : 调用任意 OpenAI 兼容接口（OpenAI / DeepSeek / 通义 / 本地 vLLM 等）
BACKEND = os.environ.get("HELLOAGENT_BACKEND", "mock").strip().lower()

# ---- OpenAI 兼容接口配置（仅 BACKEND=openai 时生效） ----
API_KEY = os.environ.get("HELLOAGENT_API_KEY", "")
BASE_URL = os.environ.get("HELLOAGENT_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("HELLOAGENT_MODEL", "gpt-4o-mini")

# ---- 安全阀 ----
# 单个任务里最多允许几步推理，防止死循环烧钱、也防止 Mock 卡死
MAX_STEPS = int(os.environ.get("HELLOAGENT_MAX_STEPS", "6"))

# ---- 记忆 ----
# 只保留最近 N 条消息，防止上下文无限膨胀
MAX_HISTORY = int(os.environ.get("HELLOAGENT_MAX_HISTORY", "12"))

# ---- 默认人格 ----
SYSTEM_PROMPT = "你是一个严谨、简洁的助手。不知道就说不知道，不要编造。"


def describe() -> str:
    """返回当前配置的一行摘要，方便排查。"""
    if BACKEND == "openai":
        return f"backend=openai  model={MODEL}  base_url={BASE_URL}"
    return "backend=mock（离线规则模拟，无需密钥）"
