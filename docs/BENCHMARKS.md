# 本地评测记录（Qwen3-ASR-1.7B vs Whisper large-v3）

测试场景：小声普通话夹杂英文技术术语（20 句、分 5 段 20–38 秒 16kHz 单声道录音），硬件为 RTX 3060 12GB。

## 核心指标对比

| 方案 | 音频版本 | 热词 / Prompt | CER（全部） | 术语命中 | 备注 |
|---|---|---|---:|---:|---|
| Whisper large-v3 | raw | 无 | 27.25% | - | 对低音量输入敏感 |
| Whisper large-v3 | norm (-20 dBFS) | 无 | 18.72% | - | 归一化后明显改善 |
| Qwen3-ASR-1.7B | raw | 无 | 11.37% | 15/21 | 对低音量不敏感 |
| Qwen3-ASR-1.7B | norm (-20 dBFS) | 无 | 10.90% | 15/21 | 默认 `language="Chinese"` |
| Qwen3-ASR-1.7B | norm (-20 dBFS) | 通用程序员热词（`context.txt`） | 9.72% | 16/21 | 日常推荐配置 |
| Qwen3-ASR-1.7B | norm (-20 dBFS) | 强针对性热词（含测试集词） | **6.87%** | 19/21 | 展示热词上限 |

## 延迟与显存（RTX 3060 12GB, bf16）

- 显存峰值：约 4.1–4.8 GB（含 CUDA context）
- HTTP 往返延迟（预热后）：
  - 5s 音频：**0.28s**
  - 15s 音频：**0.87s**
  - 30s 音频：**1.60s**
  - 实时率 RTF ≈ 0.04–0.05
- 模型下载提示：若 `hf-mirror` 卡在 xet 传输，可设置 `HF_HUB_DISABLE_XET=1` 直连 Hugging Face。
