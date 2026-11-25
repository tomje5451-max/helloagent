#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键跑完全部五课，适合第一次 clone 下来时快速确认环境正常。

    python run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LESSONS = [
    ("第 1 课 · 最小智能体", "lessons/01_minimal.py"),
    ("第 2 课 · 工具系统", "lessons/02_tools.py"),
    ("第 3 课 · ReAct 推理循环", "lessons/03_react.py"),
    ("第 4 课 · 记忆管理", "lessons/04_memory.py"),
    ("第 5 课 · 多智能体协作", "lessons/05_multi_agent.py"),
]


def main() -> int:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    failed = []
    for title, rel in LESSONS:
        path = os.path.join(HERE, rel)
        print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
        result = subprocess.run([sys.executable, path], cwd=HERE, env=env)
        if result.returncode != 0:
            failed.append(title)

    print(f"\n{'=' * 68}")
    if failed:
        print("以下课程运行失败：")
        for t in failed:
            print(f"  - {t}")
        return 1
    print("五节课全部运行成功。接下来建议按 docs/01-原理.md 的顺序读源码。")
    print("跑测试：python -m unittest discover -s tests -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
