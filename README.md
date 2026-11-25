# helloagent

> 用一个能跑起来的最小实现，把「智能体（Agent）」这五个字拆开讲清楚。

这个仓库是为了**学**而写的，不是为了用。全部代码只用 Python 标准库，**不需要 API Key、不联网**就能跑通，所以你可以放心读、放心改。

```
Agent = 大模型 + 工具 + 记忆 + 循环控制
```

少任何一项，它就只是个聊天机器人。

---

## 30 秒上手

```bash
git clone git@github.com:tomje5451-max/helloagent.git
cd helloagent

# 不用装任何依赖，直接跑
python lessons/01_minimal.py
```

> 需要 Python 3.9 或以上。没有第三方包，`pip install` 那一步可以跳过。

跑测试：

```bash
python -m unittest discover -s tests -v
```

---

## 五节课，从"能说话"到"会协作"

| 课 | 文件 | 你会看到什么 |
|---|---|---|
| 01 | [`lessons/01_minimal.py`](lessons/01_minimal.py) | 最简单的智能体：一次对话 + 一次工具调用 |
| 02 | [`lessons/02_tools.py`](lessons/02_tools.py) | 工具为什么必须写清用途，参数错了怎么办 |
| 03 | [`lessons/03_react.py`](lessons/03_react.py) | ReAct 循环：边想边做，看得到每一步轨迹 |
| 04 | [`lessons/04_memory.py`](lessons/04_memory.py) | 三种记忆的取舍，上下文为什么有成本 |
| 05 | [`lessons/05_multi_agent.py`](lessons/05_multi_agent.py) | 规划 / 执行 / 汇总，拆角色比堆一个更大的模型靠谱 |

建议顺序：**1 → 2 → 3 → 4 → 5**。每课结尾都留了"观察与思考"，那些问题比代码本身重要。

---

## 代码结构

```
helloagent/
├── helloagent/          # 核心包（可被导入）
│   ├── message.py       # 消息与对话历史 —— Agent 的"记性"
│   ├── tools.py         # 工具注册表与三个内置工具 —— Agent 的"手"
│   ├── llm.py           # 模型客户端：MockLLM / OpenAILLM —— Agent 的"脑"
│   ├── agent.py         # SimpleAgent：一次工具调用
│   ├── react.py         # ReActAgent：推理循环
│   ├── memory.py        # 无记忆 / 窗口记忆 / 摘要记忆
│   ├── multi.py         # 规划者 + 执行者 + 汇总者
│   └── config.py        # 全局配置（环境变量）
├── lessons/             # 五节可独立运行的演示
├── tests/               # 18 个单元测试
└── docs/
    ├── 01-原理.md        # 为什么智能体要长这样
    └── 02-接入真实模型.md # 换成 OpenAI / DeepSeek 要改什么
```

---

## 为什么默认用 MockLLM？

因为它把"学习"和"花钱"解耦了。

`MockLLM` 用规则模拟大模型的输出，**但它遵守和真模型一样的输入输出契约**：进 `messages`，出文本。所以：

- 你现在学的调用方式，接上真模型后一字不改；
- 调试工具、记忆、循环逻辑时，不消耗 token；
- 规则写错了，你能立刻看出是哪条规则的问题。

想接真实模型时，只需设置环境变量（见 [docs/02-接入真实模型.md](docs/02-接入真实模型.md)）：

```bash
# PowerShell
$env:HELLOAGENT_BACKEND="openai"
$env:HELLOAGENT_API_KEY="sk-..."
$env:HELLOAGENT_BASE_URL="https://api.deepseek.com/v1"
$env:HELLOAGENT_MODEL="deepseek-chat"

python lessons/01_minimal.py
```

---

## 三个我自己踩过的坑

1. **工具描述写得太抽象，模型就会乱选。**
   把 `"处理文本"` 改成 `"把英文转大写，当用户说『转大写』时用"`，命中率立刻不一样。工具的 `description` 不是注释，是提示词。

2. **ReAct 没有步数上限，就可能一直转圈。**
   `max_steps` 是保命的。而且步数用完时必须**明确说"没得出答案"**，不能编一个。

3. **记忆不是越多越好。**
   全量塞进去，每轮成本和延迟都线性上涨。工程上常见的是"最近 N 轮 + 长期摘要 + 向量检索"三件套，本仓库实现了前两件。

---

## 和 Hello-Agents 的关系

本仓库是**学习型最小实现**，思路受到 [Hello-Agents](https://github.com/datawhalechina/hello-agents)（Datawhale 社区的系统性智能体教程）启发，但代码与文档均为独立编写，体量小得多，适合先跑通、再去看那本更完整的教程。

如果你已经跑完这五课，下一步建议：

- 把 `MockLLM` 换成真实模型，观察 `trace()` 的变化；
- 给 ReAct 加上"工具调用失败后重试"；
- 把 `WindowMemory` 换成向量检索；
- 然后去读 Hello-Agents 的对应章节。

---

## License

MIT
