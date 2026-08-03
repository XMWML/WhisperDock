# WhisperDock

一个以本地优先、可整体搬迁为目标的 Whisper 语音识别工作台。它在浏览器中提供完整 WebUI：管理模型、单条和批量识别、浏览和导出结果，以及分段近实时麦克风识别。

默认识别引擎是官方 [openai/whisper](https://github.com/openai/whisper)。为兼容社区微调模型，WhisperDock 也可以通过 Transformers 加载 Hugging Face 上的 Whisper 模型。

> 数据不离开本机。模型下载、Python 环境、缓存、上传临时文件、识别结果和日志都存入 WhisperDock 自己的文件夹，不会写到 `~/.cache`、`~/.cache/huggingface` 或系统临时目录。复制整个文件夹到外置硬盘或另一台相同架构的电脑即可带走模型和历史。

## 功能

- 官方 Whisper `tiny`、`base`、`small`、`medium`、`large-v1/v2/v3`、`turbo` 模型库，可下载、删除、加载到内存、卸载。
- 自定义模型库：Hugging Face 仓库、直接 `.pt`/`.bin`/`.ckpt` 权重链接、或已放进项目 `models/` 的本地模型。
- 单条上传/麦克风录音识别，支持文字、JSON、SRT、VTT、TSV 导出。
- 后台批量识别、进度追踪、逐条结果和 CSV 汇总下载。
- 分段近实时麦克风识别。官方 Whisper 没有原生持续流式 API；此模式把音频切成可重叠短段后连续转写，适合记录而不是低延迟字幕。
- 模型加载状态、显存/内存占用与所有项目内数据路径可见。
- Whisper `transcribe()` 的语言、任务、温度回退、beam search、候选数、时间戳、提示词、静音判定、幻觉阈值等参数都有 GUI 控件，并能保存为预设。

## 快速开始

WhisperDock 支持 macOS 和 Linux，要求 Python 3.10 至 3.13。首次启动会在项目目录创建 `.venv/` 并安装依赖。

```bash
cd /path/to/WhisperDock
chmod +x bootstrap.sh run.sh start-macos.command start-linux.sh
./run.sh
```

`run.sh` 是跨平台的前台入口。macOS 可以双击 `start-macos.command`，Linux 可以运行 `./start-linux.sh`；两者都会保持当前命令行窗口打开，实时显示服务日志，并自动打开浏览器。按 `Ctrl+C` 停止服务，不再需要单独的停止脚本。日志同时保存到 `logs/whisperdock.log`，临时 PID 保存在 `workspace/whisperdock.pid`；不会创建 LaunchAgent、systemd 服务或写入项目目录外的应用状态。Linux/macOS 都可通过 `WHISPERDOCK_PORT=9000 ./run.sh` 修改端口。

`imageio-ffmpeg` 会把兼容的 FFmpeg 二进制安装在项目虚拟环境内，通常不需要另装系统 FFmpeg。如果遇到特别少见的音频封装格式仍无法解析，可安装系统 `ffmpeg` 作为补充。

## 模型指南

在「模型」页点击「添加模型」，选择引擎后填写来源。内置模型先点击「下载」，自定义模型提交后会立即下载；下载完成后点击「加载」才会占用内存。

| 想使用的模型 | 选择的引擎 | 来源栏应填写 | 去哪里找 |
| --- | --- | --- | --- |
| 官方通用模型 | `OpenAI Whisper` | 模型大小，例如 `base` 或 `turbo` | WebUI 内置模型库；官方 [Available models](https://github.com/openai/whisper#available-models) |
| Hugging Face 微调 Whisper | `Transformers / Hugging Face` | 仓库 ID，例如 `owner/whisper-finetuned-model`，或对应 Hugging Face 链接 | [Hugging Face Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending) |
| OpenAI Whisper 权重文件 | `OpenAI Whisper` | 可直接下载的、以 `.pt`、`.bin` 或 `.ckpt` 结尾的模型文件 URL | 官方模型链接或模型发布者给出的直链；不能填网页预览页 |
| 已拥有的本地模型 | 与模型格式对应 | 项目 `models/` 下的相对路径 | 将整个 Transformers 模型目录或 `.pt`/`.bin`/`.ckpt` 文件放入项目 `models/` 后添加 |

### 微调模型示例

添加 Transformers 格式的微调模型时，选择 **Transformers / Hugging Face**，填写对应的 Hugging Face 仓库 ID：

```text
owner/whisper-finetuned-model
```

下载后的模型会进入 `models/huggingface/`。模型说明和许可请以对应的 Hugging Face 页面为准。

### 选择大小

| 大小 | 适合场景 | 大致权重 | 备注 |
| --- | --- | ---: | --- |
| `tiny` / `base` | 快速草稿、CPU | 75 MB / 142 MB | 精度和速度的入门选择 |
| `small` | 日常本地识别 | 466 MB | 常用平衡点 |
| `medium` | 更高准确率 | 1.5 GB | 内存/显存需求显著提高 |
| `large-v3` | 最高通用准确率 | 约 3 GB | 建议 GPU 或较大内存 |
| `turbo` | 较快的高质量转录 | 约 1.6 GB | 适合转录，不支持翻译任务 |

实际占用还取决于精度、设备和音频长度。CPU 使用 `base` 或 `small` 较稳妥；NVIDIA CUDA 可使用半精度；Apple Silicon 可以在设置中选择 MPS，若某个模型报兼容错误则切回 CPU。

## 参数入口

「转录」页的“高级参数”展开面板覆盖官方 Whisper `transcribe()` 的主要参数：

- 通用：语言、转录/翻译任务、设备、FP16、详细日志、初始提示词/前缀。
- 解码：温度与递增回退、beam size、best of、patience、length penalty、采样回退阈值、token 抑制、静音 token、最大初始时间戳。
- 分段：30 秒分段长度、时间范围、前文条件、初始提示词传递、温度触发后的 prompt 重置。
- 质量防护：压缩比、平均对数概率、无语音阈值、静音幻觉阈值。
- 时间轴：词级时间戳、字幕格式、前后标点、时间戳开关。

Transformers 格式模型由其自身生成配置约束；不适用的 OpenAI 参数会在页面中标记，并由后端安全忽略而不是静默报错。

## 项目内数据布局

```text
WhisperDock/
├── .venv/                # 本项目的 Python 和依赖
├── .home/                # 工具所需的本地 home，防止写入用户目录
├── cache/                # pip、Torch、Hugging Face、XDG、临时缓存
├── models/               # 已下载或导入的模型
├── workspace/            # 上传音频和临时分段
├── outputs/              # 可下载的识别结果
├── config/               # 模型库、参数预设、历史记录
└── logs/                 # 本地服务日志
```

迁移时先在 WebUI 卸载模型并退出服务，再完整复制这一层文件夹。`.venv` 中的二进制依赖通常只可在相同操作系统和 CPU 架构上直接复用；跨架构时保留 `models/`、`cache/huggingface/`、`config/`、`outputs/`，删除目标机器上的 `.venv/` 并重新运行 `./bootstrap.sh` 即可，无需重新下载模型。

## 开发与验证

```bash
./.venv/bin/python -m pytest
./.venv/bin/python -m compileall backend
```

运行服务器后，可访问 `http://127.0.0.1:8848/api/health` 检查路径隔离和依赖状态。

## 隐私与许可

WhisperDock 默认只绑定 `127.0.0.1`，不会上传音频。用户自行添加的模型、音频和导出文本只保存在本机；请遵守模型和音频内容各自的许可与隐私要求。WhisperDock 源码采用 [MIT License](LICENSE)。
