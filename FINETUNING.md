# Tacotron2 微调试跑说明

## 已验证的流程

固定划分名单 → 32 条短句训练池 / 8 条验证 → 4 次更新 → 保存模型与 Adam 状态 → 新进程恢复并更新第 5 步 → 再次启动新进程，核对全部模型张量并合成 WAV。

批量大小为 2，前 5 步实际使用训练池中的 10 条不同训练录音。32 条是准备好的样本池，并非完成了一遍训练。测试集没有用于本次试跑。

## 官方配置核对

参考文件保存在 reference/speechbrain-v1.0.3/，来源：

- https://github.com/speechbrain/speechbrain/blob/v1.0.3/recipes/LJSpeech/TTS/tacotron2/hparams/train.yaml
- https://github.com/speechbrain/speechbrain/blob/v1.0.3/recipes/LJSpeech/TTS/tacotron2/train.py
- https://huggingface.co/speechbrain/tts-tacotron2-ljspeech
- https://huggingface.co/speechbrain/tts-hifigan-ljspeech

程序核对了官方 v1.0.3 配置与本地模型推理配置中的 22 个结构参数，全部一致；本地权重严格加载成功。这确认了模型结构兼容，但推理配置没有完整记录历史预训练过程，因此不能仅凭它证明全部历史预处理设置。

沿用该版本官方训练流程：

| 项目 | 本次配置 |
|---|---|
| 采样率 | 22050 Hz |
| n_fft / win_length / hop_length | 1024 / 1024 / 256 |
| 梅尔频带 / 频率范围 | 80 / 0–8000 Hz |
| power / normalized | 1 / False |
| norm / mel_scale | slaney / slaney |
| 动态范围压缩 | log(clamp(mel, min=1e-5))，官方函数 |
| 音频读取 | soundfile float32 PCM，保持原采样率，无裁剪或峰值归一化 |
| 文字 | metadata 第三列规范化文字，再执行官方 english_cleaners |
| 字符表 | 官方 text_to_sequence，兼容 148 个符号的嵌入层 |
| 批次 | 官方 TextMelCollate，长度排序、补齐、结束门目标 |
| 损失 | 官方 Loss：解码器及 postnet 频谱 MSE + 结束门 BCE |

## 与官方从零训练配方的区别

- 官方学习率 0.001，本次微调 0.00001。
- 官方 batch_size 64，本次 2；Windows 简化为当前进程加载数据。
- 官方训练 750 epochs；本次仅 4 步，随后恢复再更新 1 步。
- 引导注意力辅助损失权重从官方 50 改为 0：已有预训练对齐，本次先验证基本微调流程。
- 不使用官方从零训练的学习率调度，保持小学习率恒定；梯度范数裁剪阈值为 1。
- 固定已有 80/10/10 名单，不调用官方默认 90/10 随机划分准备流程。
- 仅从 train 和 val 各自名单按原顺序选择 1–5 秒录音；这会产生短句偏向，仅适合作为功能试跑。
- HiFi-GAN 不参与训练，仅在重载验证时用于合成音频。
- 使用本地预训练文件加载模型；没有改动原始权重或数据。

## 文件入口

- scripts/tts_training_common.py：路径、兼容处理、预处理、批次、损失、模型加载与语音生成。
- scripts/prepare_finetune_data.py：从已有名单准备试跑 JSON，检查模型配置，保存来源与校验值。
- scripts/train_tacotron2.py：训练、验证、检查点、恢复训练和独立重载验证。
- configs/pilot.json：试跑配置。
- data_manifests/pilot/：train.json、val.json、各自 ID 名单、provenance.json。
- checkpoints/pilot/last.pt：最新第 5 步检查点。
- checkpoints/pilot/best.pt：按本次已测验证损失选出的检查点（第 4 步）。
- logs/pilot/metrics.csv：训练损失、门损失、梯度范数、验证损失及耗时。
- logs/pilot/training_report.json：初次 4 步训练结果。
- logs/pilot/resume_training_report.json：恢复第 5 步结果。
- logs/pilot/reload_verification.json：新进程重载与 WAV 验证结果。
- wav_examples/pilot_reloaded_step5.wav：重载最新检查点后生成的音频。

## 使用方式（在 a3 目录运行）

只重新加载已有检查点并合成语音：

```powershell
.\.venv\Scripts\python.exe scripts\train_tacotron2.py --verify-only
```

继续到第 6 步（会更新 pilot 的 last.pt 和日志）：

```powershell
.\.venv\Scripts\python.exe scripts\train_tacotron2.py --resume --steps 6
```

默认训练入口检测到已有检查点会停止，避免误覆盖。新实验需使用新的配置 name 和输出目录；不要拿 pilot 最后权重作为 expA 或 expB 的共同起点，两组正式实验都应从原始预训练权重开始。

检查点包含模型参数、Adam 状态、步数、最佳验证损失、配置、数据文件哈希、Python/NumPy/PyTorch CPU/GPU 随机状态。恢复时严格检查配置和数据哈希。验证阶段的 Tacotron2 prenet 仍可能使用随机 dropout，短验证集数值会波动；不应从这次验证损失变化推断音质提升。

此训练脚本为学习与功能验证脚手架，不是完整作业成果。正式实验还需扩大并固定训练子集、确定预算和评估协议、整合 Notebook，并记录 AI 辅助使用。预训练模型已使用 LJSpeech，当前测试划分不能消除预训练数据重叠，应在报告中说明。

