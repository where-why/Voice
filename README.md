# Voice TTS 项目运行指南

基于 PyTorch 的 Tacotron2 文字转语音（TTS）项目，Python 3.11。

---

## 环境要求

| 依赖 | 说明 |
|------|------|
| Python | 3.11 |
| PyTorch | 2.0 及以上 |
| torchaudio | 2.0 及以上 |
| ffmpeg | 用于读取 `.m4a` 音频，需单独安装并加入系统 PATH |

---

## 项目结构

```
Voice/
├── train_data/
│   ├── voice/          # 训练音频
│   ├── label/          # 训练文本标注
│   └── processed/      # 训练集预处理缓存
├── vel_data/
│   ├── voice/          # 验证音频
│   └── label/          # 验证文本标注
├── checkpoints/        # 模型权重
├── outputs/            # 推理输出
├── config.py           # 模型与音频超参数
├── preprocess.py       # 数据预处理
├── train.py            # 模型训练
└── infer.py            # 文字合成语音
```

**数据配对规则**：`voice/1.wav` 对应 `label/1.txt`，文件名必须一致。

---

## 快速开始

所有命令均在项目根目录 `Voice/` 下执行。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

直接运行，配置在 `preprocess.py` 顶部：

```bash
python preprocess.py
```

可修改的配置项：

```python
VOICE_DIR = ROOT / "data" / "voice"
LABEL_DIR = ROOT / "data" / "label"
CACHE_DIR = ROOT / "data" / "processed"
FORCE = False  # 改为 True 强制重新处理
```

### 3. 训练模型

直接运行，配置在 `train.py` 顶部：

```bash
python train.py
```

可修改的配置项：

```python
EPOCHS = 500
BATCH_SIZE = 1
RESUME = False  # 改为 True 从 tacotron2_latest.pt 恢复训练
USE_AMP = True  # 混合精度
```

最佳模型指标（`train.py` 顶部 `BEST_METRIC`）：

| 值 | 说明 |
|----|------|
| `mel` | val_mel_loss，仅 Mel 重建误差 |
| `mel_postnet` | val_mel_postnet_loss，PostNet 后 Mel 误差（推荐） |
| `combined` | `0.9 * mel + 0.1 * gate`，可改权重 |

每个 epoch 训练结束后在 **vel_data 验证集**上评估，并保存：
- `tacotron2_latest.pt` — 最新模型
- `tacotron2_best.pt` — 上述选定指标在验证集上**最低**的模型

长音频会按顺序切为 512 帧/段（约 6 秒），每 epoch 从头到尾遍历全部片段。

### 4. 文字合成语音

直接运行，配置在 `infer.py` 顶部：

```bash
python infer.py
```

可修改的配置项：

```python
TEXT = "乡村振兴，人才是关键。"
CHECKPOINT = CHECKPOINT_DIR / "tacotron2_best.pt"  # 或 tacotron2_latest.pt
OUTPUT = OUTPUT_DIR / "output.wav"
```

---

## 完整流程

```bash
cd D:\pycharmmax\Voice
pip install -r requirements.txt
python preprocess.py
python train.py
python infer.py
```

---

## 常见问题

**读取 m4a 报错**：未安装 ffmpeg 或未加入 PATH，安装后重启终端。

**合成质量差**：训练样本较少时需扩充 `train_data/`，验证集放 `vel_data/`。

**GPU 不可用**：确认安装了 CUDA 版 PyTorch，`import torch; print(torch.cuda.is_available())` 应为 `True`。

---

## 配置说明

| 文件 | 内容 |
|------|------|
| `train.py` 顶部 | 数据路径、训练轮数、学习率、是否恢复训练 |
| `preprocess.py` 顶部 | 数据路径、是否强制重新处理 |
| `infer.py` 顶部 | 合成文本、模型路径、输出路径 |
| `config.py` | 模型结构、音频采样率、Mel 维度等 |
