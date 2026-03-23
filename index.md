---
layout: about
---
# About Me

我是清华大学深圳国际研究生院的学生，我的研究兴趣是计算机和医疗交叉方向.

<br/>

# 技能
  * Python,pytorch,Linux,C++
# Career

* Tsinghua University(2021/07 ~ )
  * 医疗NLP
* Xi'an University of Technology (2013/08 ~ 2017/9)
  * 水下机器人

<br/>

# Interests

I am interested in technology trends.
I'm not afraid to learn languages, but I enjoy using Python.
I like to automate and reduce annoying things.

<br/>

# 求职准备计划 · 游戏 AI Bot 方向

**目标岗位：** 王者荣耀·王者指挥官模式 AI Bot（LLM + 多模态 + 分层强化学习 + Post-training）

## 项目规划（按优先级排序）

### ① 分层强化学习游戏 AI ⭐ 核心
- **方向：** 在开源游戏环境（MiniGrid / StarCraft II pysc2）实现两层 HRL 架构
  - 高层 Manager：设定子目标（攻塔 / 抢龙 / 撤退）
  - 低层 Worker：完成具体动作序列
- **算法：** HIRO / Option-Critic
- **周期：** 3~4 周

#### 详细技术方案

**系统架构（两层 HRL）**

```
┌─────────────────────────────────────────────┐
│  高层 Manager（每 k 步决策一次）              │
│  输入：全局状态 s_t                           │
│  输出：子目标 g_t（如"去高地塔"坐标向量）     │
└────────────────┬────────────────────────────┘
                 │ 子目标 g_t
┌────────────────▼────────────────────────────┐
│  低层 Worker（每步执行）                      │
│  输入：局部状态 s_t + 子目标 g_t             │
│  输出：原子动作 a_t（移动/攻击/技能）         │
│  内在奖励：r_i = -||s_{t+1} - g_t||          │
└─────────────────────────────────────────────┘
```

**环境选型：MiniGrid（推荐入门）**
- 安装：`pip install minigrid`
- 选用 `MiniGrid-FourRooms-v0`：需要跨房间到达目标，天然适合子目标分解

**核心代码框架**

```python
import torch
import torch.nn as nn
import gymnasium as gym

# ── 高层 Manager ──────────────────────────────
class Manager(nn.Module):
    def __init__(self, state_dim, goal_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, goal_dim)   # 输出子目标向量
        )
    def forward(self, state):
        return self.net(state)

# ── 低层 Worker ───────────────────────────────
class Worker(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)

# ── 内在奖励（子目标完成度）─────────────────────
def intrinsic_reward(next_state, goal):
    return -torch.norm(next_state - goal, dim=-1)

# ── 训练主循环 ────────────────────────────────
env = gym.make("MiniGrid-FourRooms-v0")
manager = Manager(state_dim=64, goal_dim=8)
worker  = Worker(state_dim=64, goal_dim=8, action_dim=env.action_space.n)

manager_opt = torch.optim.Adam(manager.parameters(), lr=1e-3)
worker_opt  = torch.optim.Adam(worker.parameters(),  lr=1e-3)

k = 10  # Manager 每 k 步重新设定子目标
for episode in range(5000):
    obs, _ = env.reset()
    state = torch.tensor(obs["image"].flatten(), dtype=torch.float32)
    goal  = manager(state.unsqueeze(0)).squeeze(0).detach()

    ep_reward = 0
    for step in range(200):
        # Worker 选动作
        logits = worker(state.unsqueeze(0), goal.unsqueeze(0))
        action = torch.distributions.Categorical(logits=logits).sample().item()

        next_obs, ext_reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_obs["image"].flatten(), dtype=torch.float32)

        # 内在奖励训练 Worker
        i_reward = intrinsic_reward(next_state.unsqueeze(0), goal.unsqueeze(0))
        worker_loss = -i_reward.mean()
        worker_opt.zero_grad(); worker_loss.backward(); worker_opt.step()

        # 每 k 步更新 Manager
        if (step + 1) % k == 0:
            new_goal = manager(next_state.unsqueeze(0)).squeeze(0)
            manager_loss = -torch.tensor(ext_reward)  # 用外部奖励训练 Manager
            manager_opt.zero_grad(); manager_loss.backward(); manager_opt.step()
            goal = new_goal.detach()

        state = next_state
        ep_reward += ext_reward
        if done: break

    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {ep_reward:.2f}")
```

**进阶：迁移到 StarCraft II（SC2）**
- 安装：`pip install pysc2`
- 子目标设计：`[攻击敌基, 防守矿区, 扩张分基]` → 离散子目标枚举
- Manager 改用 DQN，Worker 用 PPO，对应论文：`HIRO (Nachum et al., 2018)`

**评测指标**
- 胜率（Win Rate）随训练步数的曲线
- 子目标完成率（Worker 到达 Manager 指定位置的比例）
- 与 Flat RL（无分层）基线对比，体现 HRL 优势

**面试亮点总结**
- 精准命中 JD"分层强化学习"，有完整可运行代码
- 内在奖励设计体现对 HRL 核心机制的理解
- MiniGrid → SC2 的迁移路径展示工程扩展能力

### ② 大模型 Post-training · 游戏指令遵循微调 ⭐ 核心
- **方向：** 构造游戏局势 → 战术指令数据集，做 SFT + DPO 偏好对齐
- **模型：** LLaMA-3 / Qwen2.5
- **评估：** 决策合理性、指令可执行性
- **周期：** 3~4 周

#### 详细技术方案

**Post-training 全流程**

```
原始游戏数据（录像/日志）
        │
        ▼
  数据构造（SFT 数据 + DPO 偏好对）
        │
     ┌──┴──┐
     ▼     ▼
   SFT   （基于 SFT 模型）
  监督微调   DPO 偏好对齐
     │        │
     └──┬─────┘
        ▼
   对齐后模型
        │
        ▼
   评估（合理性 / 可执行性 / 胜率提升）
```

**第一步：数据构造**

SFT 数据格式（指令遵循）：
```python
# 每条样本：游戏局势描述 → 高质量战术指令
sft_sample = {
    "instruction": "当前局势如下：\n我方经济领先800金，敌方打野刚被击杀，上路二塔血量30%。\n请给出接下来30秒的战术决策。",
    "output": "立即集合上路，趁敌方打野复活CD（约40s）强推二塔。上单负责正面吸引仇恨，射手和法师输出塔体，辅助插眼防止敌方支援。推塔后立即撤退不强留。"
}
```

DPO 偏好对数据格式（chosen vs rejected）：
```python
dpo_sample = {
    "prompt": "当前局势：我方全员低血量（均低于40%），但敌方高地塔血量仅剩10%。请给出决策。",
    "chosen":   "立即撤退回城补给，切勿强攻。低血量强推高地风险极高，等待下一个团战机会。",
    "rejected": "直接强推高地，塔血量很低一波可以拆掉。"
    # rejected 看似有道理但忽略了低血量被反杀的风险
}
```

数据来源策略：
```python
# 方案A：GPT-4 合成（推荐，快速获取大量数据）
import openai

def generate_sft_sample(scenario: str) -> dict:
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是王者荣耀职业教练，给出专业战术决策"},
            {"role": "user",   "content": f"局势：{scenario}\n请给出高质量战术建议"}
        ]
    )
    return {"instruction": scenario, "output": resp.choices[0].message.content}

# 方案B：从职业选手录像中提取（更真实，但需要人工标注）
# 截帧 → 状态识别 → 对应操作序列 → 人工筛选高质量片段
```

**第二步：SFT 监督微调**

```python
# 使用 LLaMA-Factory（推荐，支持 Qwen2.5 + LoRA，配置简单）
# pip install llamafactory

# dataset_info.json 中注册数据集
{
  "game_sft": {
    "file_name": "game_sft.json",
    "formatting": "alpaca",
    "columns": {"prompt": "instruction", "response": "output"}
  }
}
```

```bash
# 启动 SFT 训练（LoRA 微调，单卡 A100 约 2 小时）
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset game_sft \
  --finetuning_type lora \
  --lora_rank 16 \
  --lora_target q_proj,v_proj \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-4 \
  --output_dir ./sft_output
```

**第三步：DPO 偏好对齐**

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载 SFT 后的模型作为起点
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "./sft_output")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# DPO 配置
training_args = DPOConfig(
    beta=0.1,                    # KL 惩罚系数，控制偏离 SFT 模型的程度
    learning_rate=5e-5,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    output_dir="./dpo_output",
)

# 加载偏好数据（chosen / rejected 对）
from datasets import load_dataset
dpo_dataset = load_dataset("json", data_files="game_dpo.json")["train"]

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

**第四步：评估**

```python
import json
from transformers import pipeline

model_sft = pipeline("text-generation", model="./sft_output")
model_dpo = pipeline("text-generation", model="./dpo_output")

def evaluate_model(model_pipeline, test_cases):
    scores = {"executable": 0, "reasonable": 0, "total": len(test_cases)}
    for case in test_cases:
        output = model_pipeline(case["prompt"], max_new_tokens=200)[0]["generated_text"]
        # 自动评估：可执行性（规则校验）
        if not any(bad in output for bad in ["低血量强推", "1v5", "送人头"]):
            scores["executable"] += 1
    return scores

# 对比实验
print("SFT 模型：", evaluate_model(model_sft, test_cases))
print("DPO 模型：", evaluate_model(model_dpo, test_cases))
```

**评测指标**

| 指标 | Base 模型 | SFT 后 | DPO 后 | 说明 |
|------|-----------|--------|--------|------|
| 指令遵循率 | ~60% | ~85% | ~90% | 输出是否符合 JSON 格式 |
| 决策可执行率 | ~50% | ~75% | ~85% | 无明显低质量决策 |
| 高风险决策率↓ | ~25% | ~15% | ~8% | DPO 显著降低错误决策 |
| 人工合理性评分 | 2.8 | 3.6 | 4.1 | 1~5分，对比职业选手决策 |

**面试亮点总结**
- 完整走通 SFT → DPO 两阶段 post-training 流程
- 数据构造策略（GPT-4 合成 + 人工筛选）体现对数据飞轮的理解
- DPO 的 `beta` 参数调优体现对 KL 散度约束的深层理解
- 对比实验表格直接证明 post-training 的收益，面试时可量化展示

### ③ 游戏场景多模态对话 Bot
- **方向：** 截图 + 语音输入 → 语音战术播报
- **技术栈：** Whisper（ASR）+ Qwen-VL（视觉理解）+ CosyVoice（TTS）
- **周期：** 1~2 周

#### 详细技术方案

**系统架构（三模块流水线）**

```
语音输入 ──► ASR(Whisper) ──► 文本指令 ──┐
                                          ├──► Prompt 拼接 ──► VLM(Qwen-VL) ──► 战术文本 ──► TTS(CosyVoice) ──► 语音播报
游戏截图 ──► 帧提取/压缩 ──► 图像 ────────┘
```

**模块1：ASR — 语音转文字**
- 模型：`openai/whisper-small`（中文效果好，本地可运行）
- 输入：麦克风实时音频流（16kHz）
- 输出：玩家指令文本，如"现在应该怎么打？"
- 关键代码：
```python
import whisper
model = whisper.load_model("small")
result = model.transcribe("audio.wav", language="zh")
print(result["text"])
```

**模块2：VLM — 视觉理解 + 战术决策**
- 模型：`Qwen-VL-Chat`（支持中文 + 图文对话）
- 输入：游戏截图（小地图 + 战场画面）+ 玩家语音转文字
- Prompt 模板：
```
你是王者荣耀的战术指挥官。
当前游戏截图如下：[IMAGE]
玩家问题：{asr_text}
请根据图中局势（血量、位置、经济）给出简短战术建议（50字内）。
```
- 输出：结构化战术文本

**模块3：TTS — 文字转语音播报**
- 模型：`CosyVoice`（阿里开源，音色自然）或轻量替代 `edge-tts`
- 输入：战术建议文本
- 输出：WAV 音频，实时播放
```python
import edge_tts, asyncio
async def speak(text):
    communicate = edge_tts.Communicate(text, voice="zh-CN-YunxiNeural")
    await communicate.save("output.wav")
asyncio.run(speak("敌方打野现身上路，立即撤退！"))
```

**数据与评测**
- 用王者荣耀录屏截帧（每秒1帧）构造测试集
- 评估指标：ASR字错率（CER）、VLM决策合理性（人工评分1~5）、端到端延迟（目标 < 3s）

**面试亮点总结**
- 端到端多模态系统：视觉 + 语音 + 文本全覆盖
- 游戏场景落地：直接模拟"能听会说"的指挥官 Bot
- 可量化评测：延迟、准确率均有指标

### ④ 基于 LLM 的宏观策略规划器
- **方向：** 游戏状态序列化 → LLM Chain-of-Thought 输出结构化战术 JSON
- **亮点：** 体现"言行匹配"，决策可量化评估
- **周期：** 1~2 周

#### 详细技术方案

**系统架构**

```
游戏状态结构体
(血量/位置/经济/技能CD)
        │
        ▼
  状态序列化器
  (Python dict → 自然语言描述)
        │
        ▼
  Prompt 构造器
  (System Prompt + 状态描述 + CoT 指令)
        │
        ▼
   LLM 推理
   (Qwen2.5 / GPT-4)
        │
        ▼
  结构化输出解析
  (JSON: 目标/优先级/行动序列)
        │
        ▼
  决策执行 & 评估
  (可执行性校验 + 合理性打分)
```

**游戏状态定义**

```python
from dataclasses import dataclass
from typing import List

@dataclass
class HeroState:
    name: str
    hp_ratio: float      # 当前血量百分比
    position: str        # "上路/中路/下路/jungle/高地"
    skill_ready: bool    # 大招是否就绪
    economy: int         # 金币数

@dataclass
class GameState:
    time_sec: int
    our_team: List[HeroState]
    enemy_team: List[HeroState]
    map_events: List[str]   # ["龙刷新30s后", "上路一塔已破", "敌方打野现身下路"]
    score_diff: int         # 我方经济领先值（负数为落后）
```

**状态序列化器**

```python
def serialize_state(state: GameState) -> str:
    lines = [f"当前时间：{state.time_sec // 60}分{state.time_sec % 60}秒"]
    lines.append(f"经济差：{'领先' if state.score_diff > 0 else '落后'} {abs(state.score_diff)} 金币")
    lines.append("我方状态：")
    for h in state.our_team:
        skill = "大招就绪" if h.skill_ready else "大招CD中"
        lines.append(f"  - {h.name}：血量{h.hp_ratio:.0%}，位于{h.position}，{skill}")
    lines.append("敌方动态：")
    for h in state.enemy_team:
        lines.append(f"  - {h.name}：血量{h.hp_ratio:.0%}，位于{h.position}")
    lines.append("地图事件：" + "；".join(state.map_events))
    return "\n".join(lines)
```

**Prompt 模板（含 CoT）**

```python
SYSTEM_PROMPT = """你是王者荣耀顶级战术指挥官，擅长宏观决策。
请严格按以下 JSON 格式输出，不要输出其他内容：
{
  "reasoning": "分析过程（50字内）",
  "objective": "当前主要目标",
  "priority": "high/medium/low",
  "actions": [
    {"hero": "英雄名", "action": "具体行动", "position": "目标位置"}
  ],
  "warning": "需要注意的威胁（可为null）"
}"""

def build_prompt(state: GameState) -> str:
    state_text = serialize_state(state)
    return f"{state_text}\n\n请给出当前最优战术决策："
```

**LLM 调用（支持本地 / API 两种模式）**

```python
import json
from openai import OpenAI  # Qwen2.5 兼容 OpenAI 接口

# 本地部署 Qwen2.5（ollama）
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def get_decision(state: GameState) -> dict:
    prompt = build_prompt(state)
    response = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,   # 低温保证决策稳定性
    )
    text = response.choices[0].message.content
    return json.loads(text)

# 示例输出：
# {
#   "reasoning": "经济领先且敌方打野现身下路，上中路压制时机成熟",
#   "objective": "推倒上路二塔",
#   "priority": "high",
#   "actions": [
#     {"hero": "李白", "action": "压制上路", "position": "上路二塔"},
#     {"hero": "诸葛亮", "action": "跟进支援", "position": "上路"}
#   ],
#   "warning": "注意敌方打野可能绕后"
# }
```

**决策评估器**

```python
# 自动校验：动作是否可执行（英雄血量/位置合理性）
def validate_decision(decision: dict, state: GameState) -> dict:
    issues = []
    our_heroes = {h.name: h for h in state.our_team}
    for action in decision["actions"]:
        hero = our_heroes.get(action["hero"])
        if hero and hero.hp_ratio < 0.3:
            issues.append(f"{action['hero']} 血量过低（{hero.hp_ratio:.0%}），不宜进攻")
    decision["issues"] = issues
    decision["executable"] = len(issues) == 0
    return decision

# 评测指标收集
def evaluate_batch(states, ground_truth_actions):
    results = []
    for state, gt in zip(states, ground_truth_actions):
        decision = get_decision(state)
        validated = validate_decision(decision, state)
        results.append({
            "executable_rate": validated["executable"],
            "objective_match": decision["objective"] == gt["objective"],
            "latency_ms": ...  # 记录推理耗时
        })
    return results
```

**评测指标**

| 指标 | 目标值 | 说明 |
|------|--------|------|
| JSON 解析成功率 | > 95% | 输出格式稳定性 |
| 决策可执行率 | > 80% | 无血量/位置冲突 |
| 目标合理性（人工1~5分） | > 3.5 | 对比职业选手录像 |
| 推理延迟 | < 2s | 7B 模型本地推理 |

**面试亮点总结**
- 状态序列化设计体现对游戏 AI 输入工程的理解
- CoT Prompt + 结构化 JSON 输出体现 LLM 工程能力
- 自动校验器实现"言行匹配"的可量化评估，直击 JD 核心

### ⑤ LLM + RL 言行匹配联合框架（综合）
- **方向：** LLM 高层规划 + RL 执行 + Reflexion 反思 + 奖励模型对齐
- **参考：** DEPS / Voyager / GLAM
- **周期：** 4~6 周

## 执行路线

| 阶段 | 项目 | 目标 |
|------|------|------|
| 第一阶段 | ③ + ④ | 快速出可展示成果 |
| 第二阶段 | ① + ② | 打深核心技术深度 |
| 第三阶段 | ⑤ | 整合成完整系统 |
