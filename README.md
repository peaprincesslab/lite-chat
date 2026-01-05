# Lite Chat

一个轻量的多模态AI对话应用，支持文本对话、图像生成、图生图和图生文功能。

## ✨ 功能特性

- 💬 **文本对话**：基于 Qwen3-4B-Instruct 模型的智能对话
- 🎨 **文生图**：使用 Stable Diffusion v1.5 根据文本描述生成图像
- 🔄 **图生图**：基于参考图像生成新图像
- 📝 **图生文**：使用 BLIP 模型生成图像描述，并支持 AI 改写
- 📄 **文件处理**：支持上传多种格式文件（TXT、MD、PDF、DOCX等）进行内容分析
- ⚡ **懒加载**：模型按需加载，节省内存和启动时间
- 🚀 **本地部署**：完全本地运行，保护隐私

## 📋 系统要求

- Python 3.8+
- CUDA 11.8+ (可选，用于 GPU 加速)
- 至少 16GB RAM (推荐 32GB+)
- 至少 20GB 可用磁盘空间（用于存储模型）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

运行模型下载脚本：

```bash
python download_models.py
```

脚本支持下载以下模型：
- **Stable Diffusion v1.5** (约 6.5GB) - 文生图/图生图
- **BLIP** (约 1GB) - 图生文
- **Qwen3-4B-Instruct** (约 8GB) - 对话模型

### 3. 启动 Web UI

```bash
python run_webui.py
```

启动后，在浏览器中访问 `http://localhost:7860`

## 📁 项目结构

```
lite-chat/
├── run_webui.py          # Web UI 主程序
├── load_models.py        # 模型加载模块（懒加载）
├── download_models.py    # 模型下载脚本
├── requirements.txt      # Python 依赖
├── models/               # 模型存储目录
│   ├── Qwen3-4B-Instruct/
│   ├── stable-diffusion-v1-5-img2img/
│   └── blip-image-captioning-base/
└── doc/                  # 文档目录
```

## 🎯 使用说明

### 文本对话

1. 在"文本对话"标签页输入问题
2. 可调整生成参数（Temperature、Top-p、Top-k 等）
3. 支持上传文件进行内容分析

### 图像生成

**文生图**：
1. 切换到"图像生成" → "文生图"
2. 输入图像描述和负面提示词
3. 调整参数（分辨率、推理步数、引导强度等）
4. 点击"生成图像"

**图生图**：
1. 切换到"图像生成" → "图生图"
2. 上传参考图像
3. 输入描述和参数
4. 点击"生成图像"

**图生文**：
1. 切换到"图像生成" → "图生文"
2. 上传图像
3. 点击"生成描述"
4. 可选：使用"改写描述"功能优化描述

## ⚙️ 配置说明

### 模型路径

默认模型存储在 `./models/` 目录下：
- 对话模型：`./models/Qwen3-4B-Instruct`
- 图像模型：`./models/stable-diffusion-v1-5-img2img`
- 描述模型：`./models/blip-image-captioning-base`

可在 `load_models.py` 中修改默认路径。

### 懒加载机制

所有模型采用懒加载策略：
- 对话模型：首次发送消息时加载
- 图像模型：首次使用图像功能时加载
- 描述模型：首次使用图生文功能时加载

这样可以：
- 快速启动应用
- 节省内存（只加载需要的模型）
- 按需使用资源

## 🔧 高级配置

### 修改模型路径

编辑 `load_models.py`：

```python
DEFAULT_MODELS_DIR = "./models"
DEFAULT_CHAT_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, "Qwen3-4B-Instruct")
```

### 启用/禁用功能

编辑 `load_models.py`：

```python
OFFLINE_MODE = True  # 是否只使用本地模型
```

### 修改服务器端口

编辑 `run_webui.py` 末尾：

```python
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

## 参考

- [Qwen](https://github.com/QwenLM/Qwen) - 对话模型
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) - 图像生成模型
- [BLIP](https://github.com/salesforce/BLIP) - 图像描述模型
- [Gradio](https://github.com/gradio-app/gradio) - Web UI 框架