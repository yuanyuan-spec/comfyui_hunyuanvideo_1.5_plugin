[Read in English](./README.md)
# ComfyUI HunyuanVideo-1.5 插件

基于 **HunyuanVideo-1.5** 的 ComfyUI 插件，提供简化版和完整版节点集，方便快速上手或深度定制工作流程。

---

## ✨ 功能特性

- **简化版节点**：仅包含核心的 `HyVideo15ModelLoader` 和 `HyVideo15I2VSampler` 节点，非常适合快速测试和结果预览。
- **完整版节点**：提供更精细的节点划分，方便进行更细致的工作流程调整和替换。
- **自动下载模型**：内置自动模型下载功能；无需手动准备模型文件（也支持手动下载）。

---

## 📦 安装

### 步骤 1：安装依赖项

1. 安装 `requirements.txt` 中列出的所需库：`pip install -r requirements.txt`
2. Flash Attention：建议安装 Flash Attention 以加快推理速度并降低 GPU 内存消耗。详细的安装说明请参见 Flash Attention。

### 步骤 2：下载模型

选择以下方法之一下载模型文件（包括 `hunyuanvideo-1.5` 模型、`text_encoder` 和 `vision_encoder`）：

#### 方法 1：自动下载（推荐）

如果您希望自动下载模型，请将模型加载节点的路径设置为“None”。相应的模型将自动下载到默认目录（如果已存在，则不会重复下载）。下次运行工作流时，您可以在节点的路径选项中看到自动下载的模型。

#### 方法 2：手动下载

手动下载模型文件并将其放置在插件指定的模型目录中。详细说明请参见 [checkpoints-download.md](checkpoints-download.md)。（来自 HunyuanVideo-1.5 开源项目）。

**模型放置的目录结构如下：**

```

models/
├── clip_vision
│   └── hyvideo15
│       └── siglip
│           ├── feature_extractor
│           │   └── preprocessor_config.json
│           ├── flux1-redux-dev.safetensors
│           ├── image_embedder
│           │   ├── config.json
│           │   └── diffusion_pytorch_model.safetensors
│           ├── image_encoder
│           │   ├── config.json
│           │   └── model.safetensors
│           ├── LICENSE.md
│           ├── model_index.json
│           ├── README.md
│           └── redux.png
├── diffusion_models
│   └──hyvideo15
│       ├── 1080p_sr_distilled
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── 480p_i2v
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── 480p_i2v_distilled
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── 480p_t2v
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── 480p_t2v_distilled
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── 720p_i2v
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── 720p_i2v_distilled
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── 720p_i2v_distilled_sparse
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── 720p_sr_distilled
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── 720p_t2v
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── 720p_t2v_distilled
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       └── 720p_t2v_distilled_sparse
│           ├── config.json
│           └── diffusion_pytorch_model.safetensors
│   
├── text_encoders
│   ├── byt5-small
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   └── tokenizer_config.json
│   ├── Glyph-SDXL-v2
│   │   ├── assets
│   │   │   ├── color_idx.json
│   │   │   └── multilingual_10-lang_idx.json
│   │   └── checkpoints
│   │       └── byt5_model.pt
│   └── hyvideo15
│       └── llm
│           ├── chat_template.json
│           ├── config.json
│           ├── generation_config.json
│           ├── merges.txt
│           ├── model-00001-of-00005.safetensors
│           ├── model-00002-of-00005.safetensors
│           ├── model-00003-of-00005.safetensors
│           ├── model-00004-of-00005.safetensors
│           ├── model-00005-of-00005.safetensors
│           ├── model.safetensors.index.json
│           ├── preprocessor_config.json
│           ├── README.md
│           ├── tokenizer_config.json
│           ├── tokenizer.json
│           └── vocab.json
│       
├── upscale_models
│   └── hyvideo15
│       ├── 1080p_sr_distilled
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       └── 720p_sr_distilled
│           ├── config.json
│           └── diffusion_pytorch_model.safetensors
│   
└── vae
    └── hyvideo15
        ├── config.json
        └── diffusion_pytorch_model.safetensors
        
```

### 步骤 3：导入工作流程

1. 将提供的示例工作流程文件（例如 `simplified_I2V_workflow.json`）导入 ComfyUI。
2. 调整必要的参数，例如选择模型路径和加载图像。
3. 根据需要调整参数或替换节点（完整节点集允许更灵活的调整）。

---

## 🧩 节点说明

### 简化节点

- `HyVideo15ModelLoader`：加载混元视频-1.5 模型。
- `HyVideo15I2VSampler`：执行视频生成推理。

### 完整节点

除了简化功能外，完整节点集还包含以下拆分节点：

- `HyVideoTextEncode`：文本编码器。
- `HyVideoVisionEncode`：图像编码器。
- 更多详情请参考示例工作流程。

---

## 🛠 使用技巧

- 初次使用时，建议先使用**简化工作流程**，以便快速验证结果。
- 如果需要自定义生成逻辑（例如，替换编码器、调整帧序列），请切换到**完整节点**，以便进行更灵活的组装。
- 确保网络连接稳定，以便自动下载。如果下载失败，请检查路径或手动下载模型。

---

## ❓ 常见问题

**问：如果自动下载失败该怎么办？** 答：检查您的网络连接，或手动下载模型并将其放置在 `models/` 目录下的相应子目录中。

**问：如何在简化版和完整版之间切换？** 答：将相应的工作流程文件（例如，`simplified_I2V_workflow.json` 或 `complete_I2V_workflow.json`）导入 ComfyUI。节点按版本分组。

---

## 📄 许可协议

本插件基于混元视频-1.5 模型。请遵守原模型的相关许可协议。
