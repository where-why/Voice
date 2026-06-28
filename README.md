# Voice TTS (Tacotron2 + THCHS-30)

## 项目结构

```
Voice/
├── config.py           # 路径与模型配置
├── preprocess.py       # 预处理
├── train.py            # 训练
├── infer.py            # 推理
├── data_utils/
│   ├── audio.py        # 音频 / Mel / Griffin-Lim
│   ├── text.py         # 词表与文本编码
│   └── dataset.py      # THCHS-30 扫描与 Dataset
├── models/
│   ├── tacotron2.py    # 模型
│   └── loss.py         # 损失函数
└── data/
    ├── data_thchs30/   # THCHS-30 原始数据
    └── processed/      # 预处理缓存（自动生成）
```

## 使用

```bash
pip install -r requirements.txt
python preprocess.py
python train.py
python infer.py
```

## 训练配置（train.py）

- `BEST_METRIC`: `mel` | `mel_postnet` | `combined`
- `BATCH_SIZE`, `EPOCHS`, `RESUME`, `USE_AMP`

## 输出

- `checkpoints/tacotron2_latest.pt`
- `checkpoints/tacotron2_best.pt`（验证集指标最优）
- `outputs/output.wav`
