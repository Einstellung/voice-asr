# voice-asr：本地多模态大模型语音输入后端（Qwen3-ASR-1.7B + fcitx5-vinput）

面向 Linux 桌面端（`fcitx5` + [fcitx5-vinput](https://github.com/xifan2333/fcitx5-vinput)）的纯本地语音识别服务：使用 **`Qwen/Qwen3-ASR-1.7B-hf`**（`transformers` 原生 `AutoModelForMultimodalLM`）提供 OpenAI 兼容的 `/v1/audio/transcriptions` 接口，支持低音量说话人声归一化、热词上下文热重载、静音过滤与可选语音语料留存。

## 整体链路

```text
长按右 Alt 录音 (fcitx5-vinput)
  └─> vinput VAD + 音频归一化
        └─> provider.openai-compatible.batch (stdin PCM -> WAV POST)
              └─> voice-asr (127.0.0.1:8765, Qwen3-ASR-1.7B, bf16)
                    └─> 文本上屏（可选接 vinput LLM 场景润色 / 命令模式）
```

## 特性

- **纯本地离线推理**：模型加载后常驻 GPU（bf16 约占 4.1–4.8 GB 显存），配合 `HF_HUB_OFFLINE=1` 不联网即可工作。
- **低音量友好**：参考 [audio_utils.py](audio_utils.py)，仅对高于 `-50 dBFS` 的有声片段计算 RMS 并归一化到 `-20 dBFS`（最大增益 +40 dB，峰值限幅 ±0.99），避免小声说话漏字。
- **空输入 / 静音拦截**：短于 `0.3s` 或有声片段不足 `0.1s` 的录音直接返回空字符串，防止大模型产生幻觉。
- **热词表热重载**：修改 [context.txt](context.txt) 后无需重启服务，下一次请求自动按文件 mtime 重新加载。
- **中英混说效果好**：RTX 3060 12GB 实测 5s 音频延迟约 0.28s（RTF ≈ 0.04–0.05），详见 [docs/BENCHMARKS.md](docs/BENCHMARKS.md)。
- **可选语料留存**：保存 16kHz 单声道原始音频与识别元数据（`records.jsonl` 含 `corrected_text` 字段），便于后续微调或评测（默认不入 Git）。

## 目录结构

```text
voice-asr/
├── app.py                        # FastAPI + Qwen3-ASR-1.7B 推理服务
├── audio_utils.py                # 16kHz 解码、静音检测与响度归一化
├── context.txt                   # 术语表 / 上下文 prompt（支持热重载）
├── pyproject.toml / uv.lock      # Python 3.12 锁定依赖
├── requirements.txt              # pip 等价依赖列表
├── systemd/voice-asr.service     # systemd --user 服务模板
├── examples/
│   ├── vinput-config.json        # fcitx5-vinput 脱敏配置模板
│   └── fcitx5-vinput.conf        # fcitx5 按键配置示例（Alt_R / Control_R / Shift_R）
├── tests/
│   ├── test_audio_utils.py       # 音频处理离线单元测试
│   └── smoke_test.py             # HTTP 接口冒烟测试
└── docs/BENCHMARKS.md            # 本地评测对比记录
```

## 快速开始

### 1. 安装依赖并准备模型

需要 Python 3.12–3.13 与 NVIDIA GPU（约 5 GB 可用显存）：

```bash
uv sync
# 首次运行需从 Hugging Face 缓存或下载 Qwen/Qwen3-ASR-1.7B-hf（revision bcd2b5b7...）
.venv/bin/python app.py
```

若下载模型时遇到传输卡死，可尝试 `HF_HUB_DISABLE_XET=1`。确认服务就绪：

```bash
curl -s http://127.0.0.1:8765/health
```

### 2. 安装并配置 fcitx5-vinput（Ubuntu 24.04 示例）

```bash
sudo add-apt-repository ppa:xifan2333/ppa
sudo apt update
sudo apt install fcitx5-vinput-lite

systemctl --user enable --now vinput-daemon.service
fcitx5 -r
```

添加 OpenAI 兼容 provider 并参考 [examples/vinput-config.json](examples/vinput-config.json) 修改 `~/.config/vinput/config.json`：

```bash
vinput provider add provider.openai-compatible.batch
vinput provider use provider.openai-compatible.batch
```

关键注意事项：
- **`no_proxy` / `NO_PROXY`**：`vinput` 的批处理脚本基于 Python `urllib`，若系统设置了 HTTP 代理，必须在 provider `env` 中加 `"no_proxy": "127.0.0.1,localhost"`。
- **`VINPUT_ASR_API_KEY`**：`openai-compatible/batch` 脚本会校验该字段非空，填 `"local"` 即可，本服务不校验具体值。
- **按键参考**：见 [examples/fcitx5-vinput.conf](examples/fcitx5-vinput.conf)（长按 `Alt_R` 说话、短按切换；`Control_R` 命令模式；`Shift_R` 打开命令面板）。
- **可选 LLM 二次处理**：当前示例 `llm.providers` 为空、默认走 `__raw__` 原文上屏；如需接文本大模型做标点润色或语音改写，可在 `vinput` 的 `llm` 与 `scenes` 字段中配置兼容 OpenAI 的文本模型。

改完后重启 daemon：

```bash
systemctl --user restart vinput-daemon.service
```

### 3. 配置 systemd 用户常驻服务

```bash
mkdir -p ~/.config/systemd/user
cp systemd/voice-asr.service ~/.config/systemd/user/
# 编辑 ~/.config/systemd/user/voice-asr.service，将 WorkingDirectory 和 ExecStart 指向你的仓库路径
systemctl --user daemon-reload
systemctl --user enable --now voice-asr.service
journalctl --user -u voice-asr -f
```

需要释放显存跑其他训练/推理任务时：

```bash
systemctl --user stop voice-asr.service
```

## 环境变量说明

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VOICE_ASR_HOST` / `VOICE_ASR_PORT` | `127.0.0.1` / `8765` | 监听地址与端口 |
| `VOICE_ASR_MODEL` / `VOICE_ASR_MODEL_REVISION` | `Qwen/Qwen3-ASR-1.7B-hf` / `bcd2b5b7...` | Hugging Face 模型与锁定版本 |
| `VOICE_ASR_DEVICE` | `cuda:0` | 推理设备 |
| `VOICE_ASR_LANGUAGE` | `Chinese` | 默认语言 |
| `VOICE_ASR_CONTEXT_FILE` | `./context.txt` | 热词/上下文文件 |
| `VOICE_ASR_SAVE_CORPUS` | `1` | 语料留存开关（`0` 关闭） |
| `VOICE_ASR_CORPUS_DIR` | `./corpus` | 语料保存根目录 |
| `VOICE_ASR_CORPUS_FORMAT` | `flac` | `flac` 或 `wav` |
| `VOICE_ASR_TARGET_DBFS` / `VOICE_ASR_FLOOR_DBFS` | `-20` / `-50` | 归一化目标电平与有效语音门限 |
| `VOICE_ASR_MIN_DURATION` / `VOICE_ASR_MIN_VOICED` | `0.3` / `0.1` | 过短/静音拦截阈值（秒） |
| `VOICE_ASR_MAX_NEW_TOKENS` | `1024` | 最大生成 token 数 |
| `VOICE_ASR_WARMUP` | `1` | 启动时是否空跑一次推理预热 |

## 测试

```bash
# 离线音频处理单元测试
.venv/bin/python -m unittest tests/test_audio_utils.py

# HTTP 服务冒烟测试（需先启动服务）
.venv/bin/python tests/smoke_test.py http://127.0.0.1:8765
```

## 隐私与开源提示

- `corpus/` 目录包含本人语音录音与转写记录，已加入 `.gitignore`，切勿提交到公开仓库。
- `examples/` 中已移除本地用户名路径及密钥信息，提交前仍建议用 `git diff` 复核。

## License

[MIT](LICENSE)
