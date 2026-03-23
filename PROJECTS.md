# 游戏 AI Bot 求职项目规划

> 目标岗位：王者荣耀·王者指挥官模式 AI Bot
> 技术方向：LLM + 多模态 + 分层强化学习 + Post-training

---

## 项目总览

| # | 项目名 | 核心技术 | 周期 | 优先级 |
|---|--------|---------|------|--------|
| ① | 分层强化学习游戏 AI | HIRO / Option-Critic / PPO | 3~4 周 | ⭐⭐⭐⭐⭐ |
| ② | Post-training 游戏指令微调 | SFT + DPO / LLaMA-Factory / trl | 3~4 周 | ⭐⭐⭐⭐⭐ |
| ③ | 多模态对话 Bot | Whisper + Qwen-VL + CosyVoice | 1~2 周 | ⭐⭐⭐⭐ |
| ④ | LLM 宏观策略规划器 | Chain-of-Thought + 结构化 JSON | 1~2 周 | ⭐⭐⭐⭐ |
| ⑤ | LLM + RL 言行匹配联合框架 | Reflexion + 奖励建模 + GoalPPO | 4~6 周 | ⭐⭐⭐ |

**执行路线：**
```
第一阶段（快速出成果）：③ + ④
第二阶段（打深技术深度）：① + ②
第三阶段（整合终极系统）：⑤
```

---

## 项目① 分层强化学习游戏 AI

### 架构

```
高层 Manager（每 k 步）：全局状态 → 子目标向量（攻塔/抢龙/撤退）
        ↓
低层 Worker（每步）：局部状态 + 子目标 → 原子动作
        ↓
内在奖励：r_i = -||next_state - goal||
```

### 环境

- 入门：`MiniGrid-FourRooms-v0`（`pip install minigrid`）
- 进阶：StarCraft II `pysc2`

### 关键代码

```python
class Manager(nn.Module):
    def __init__(self, state_dim, goal_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, goal_dim)
        )
    def forward(self, state):
        return self.net(state)

class Worker(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, state, goal):
        return self.net(torch.cat([state, goal], dim=-1))

def intrinsic_reward(next_state, goal):
    return -torch.norm(next_state - goal, dim=-1)
```

### 评测指标

| 指标 | 说明 |
|------|------|
| 胜率（Win Rate） | 随训练步数的收敛曲线 |
| 子目标完成率 | Worker 到达 Manager 指定位置的比例 |
| vs Flat RL 基线 | 体现分层结构的优势 |

---

## 项目② Post-training 游戏指令微调

### 流程

```
原始数据 → SFT 数据构造 → LoRA 监督微调 → DPO 偏好对齐 → 评估对比
```

### 数据格式

**SFT 样本：**
```json
{
  "instruction": "我方经济领先800金，敌方打野刚死，上路二塔血量30%。给出战术。",
  "output": "立即集合上路强推二塔，上单吸收仇恨，射手法师输出，推后立即撤退。"
}
```

**DPO 偏好对：**
```json
{
  "prompt": "我方全员血量低于40%，但敌方高地塔仅剩10%血。给出决策。",
  "chosen":   "立即撤退回城，切勿强攻，等待下一个团战机会。",
  "rejected": "直接强推高地，塔血量很低一波可以拆掉。"
}
```

### 训练命令

```bash
# SFT（LLaMA-Factory + LoRA）
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset game_sft \
  --finetuning_type lora \
  --lora_rank 16 \
  --num_train_epochs 3 \
  --output_dir ./sft_output

# DPO（trl）
# 见详细方案代码
```

### 评测对比

| 指标 | Base | SFT | DPO |
|------|------|-----|-----|
| 指令遵循率 | ~60% | ~85% | ~90% |
| 决策可执行率 | ~50% | ~75% | ~85% |
| 高风险决策率↓ | ~25% | ~15% | ~8% |
| 人工合理性评分 | 2.8 | 3.6 | 4.1 |

---

## 项目③ 多模态对话 Bot

### 架构

```
语音输入 → Whisper(ASR) → 文本指令 ─┐
                                     ├→ Qwen-VL → 战术文本 → CosyVoice(TTS) → 语音播报
游戏截图 → 帧提取/压缩 → 图像 ────────┘
```

### 安装

```bash
pip install openai-whisper edge-tts
# Qwen-VL 可用阿里云 API，无需本地 GPU
```

### Prompt 模板

```
你是王者荣耀的战术指挥官。
当前游戏截图如下：[IMAGE]
玩家问题：{asr_text}
请根据图中局势（血量、位置、经济）给出简短战术建议（50字内）。
```

### 评测指标

| 指标 | 目标值 |
|------|--------|
| ASR 字错率（CER） | < 5% |
| VLM 决策合理性（1~5） | > 3.5 |
| 端到端延迟 | < 3s |

---

## 项目④ LLM 宏观策略规划器

### 核心设计

游戏状态结构化 → 自然语言序列化 → CoT Prompt → JSON 输出 → 可执行性校验

```python
SYSTEM_PROMPT = """你是王者荣耀顶级战术指挥官。
请严格按以下 JSON 格式输出：
{
  "reasoning": "分析过程（50字内）",
  "objective": "当前主要目标",
  "priority": "high/medium/low",
  "actions": [{"hero": "英雄名", "action": "具体行动", "position": "目标位置"}],
  "warning": "需要注意的威胁（可为null）"
}"""
```

### 本地运行（无需 GPU 服务器）

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
# 兼容 OpenAI 接口，直接调用
```

### 评测指标

| 指标 | 目标值 |
|------|--------|
| JSON 解析成功率 | > 95% |
| 决策可执行率 | > 80% |
| 人工合理性评分（1~5） | > 3.5 |
| 推理延迟 | < 2s |

---

## 项目⑤ LLM + RL 言行匹配联合框架

### 三层闭环架构

```
Layer 1  LLM 规划层    →  自然语言目标 g_text
                                ↓ Encode → g_embed
Layer 2  RL 执行层     →  动作序列（含言行匹配奖励）
                                ↓ 轨迹摘要
Layer 3  Reflexion 层  →  反思文本 → 更新 Memory → 注入下轮规划
```

### 言行匹配奖励（核心创新）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def alignment_reward(g_text: str, trajectory_summary: str, weight=0.3) -> float:
    g_embed = encoder.encode(g_text,             convert_to_tensor=True)
    t_embed = encoder.encode(trajectory_summary, convert_to_tensor=True)
    sim = torch.nn.functional.cosine_similarity(g_embed, t_embed, dim=0)
    return weight * sim.item()
```

### 评测指标

| 指标 | 无 Reflexion | 有 Reflexion |
|------|-------------|-------------|
| 言行匹配率 | ~45% | ~70% |
| 平均环境奖励 | baseline | +15~25% |
| 收敛速度 | 2000 ep | ~1400 ep |
| 目标失效率↓ | ~35% | ~18% |

---

## 技术依赖汇总

```bash
# 强化学习
pip install minigrid gymnasium torch

# 大模型微调
pip install llamafactory trl peft transformers datasets

# 多模态
pip install openai-whisper edge-tts sentence-transformers

# LLM 本地部署
curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen2.5:7b

# 通用
pip install openai torch numpy
```

---

## 面试故事线

```
③ 能听会说（多模态感知）
    ↓ 感知结果 → 游戏状态
④ LLM 策略规划（宏观决策）
    ↓ 微调让决策更准
② Post-training（SFT + DPO 对齐）
    ↓ 分层执行落地
① 分层强化学习（HRL 执行）
    ↓ 全部整合 + 言行对齐
⑤ LLM + RL 联合框架（终极系统）
```

每个项目独立可展示，五个项目串成完整系统，覆盖 JD 全部技术要求。
