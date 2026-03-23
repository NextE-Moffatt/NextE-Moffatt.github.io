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

### ② 大模型 Post-training · 游戏指令遵循微调 ⭐ 核心
- **方向：** 构造游戏局势 → 战术指令数据集，做 SFT + DPO 偏好对齐
- **模型：** LLaMA-3 / Qwen2.5
- **评估：** 决策合理性、指令可执行性
- **周期：** 3~4 周

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
