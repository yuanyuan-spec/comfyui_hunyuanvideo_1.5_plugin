[中文文档](./README_CN.md)
# ComfyUI HunyuanVideo-1.5 Plugin

A ComfyUI plugin based on **HunyuanVideo-1.5**, offering both simplified and complete node sets for quick usage or deep workflow customization.

---

## ✨ Features

- **Simplified Nodes**: Includes only the core `HyVideo15ModelLoader` and `HyVideo15I2VSampler` nodes, ideal for quick testing and result preview.
- **Complete Nodes**: Provides more finely split nodes for detailed workflow adjustments and replacements.
- **Auto-Download Models**: Built-in automatic model download; no need to manually prepare model files (manual download also supported).

---

## 📦 Installation

### Step 1: Install Dependencies
1. Install required libraries from `requirements.txt`:  `pip install -r requirements.txt`
2. Flash Attention: It's recommended to install Flash Attention for faster inference and reduced GPU memory consumption. Detailed installation instructions are available at Flash Attention.
### Step 2: Download Models
Choose one of the following methods to download the model files (including `hunyuanvideo-1.5` model, `text_encoder`, and `vision_encoder`):

#### Method 1: Auto-Download (Recommended)
Enable the **Auto-Download** option in the plugin when running a workflow. **Models will be automatically downloaded to the default path.When using the model's auto-download feature, please set the path of the model loading node to "None.**" The corresponding model will be automatically downloaded to the default directory (if it already exists, it will not be downloaded again). The next time you run the workflow, you can see the auto-downloaded model in the node's path options.

#### Method 2: Manual Download
Manually download the model file and place it in the model directory specified by the plugin. For detailed instructions, please refer to [checkpoints-download.md](checkpoints-download.md). (From the HunyuanVideo-1.5 open source project).

**The directory structure for model placement is as follows:**

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

### Step 3: Import Workflow
1. Import the provided example workflow file (e.g., `simplified_I2V_workflow.json`) into ComfyUI.
2. Adjust necessary parameters, such as selecting the model path and loading the image.
3. Tweak parameters or replace nodes as needed (the complete node set allows for more flexible adjustments).

---

## 🧩 Node Description

### Simplified Nodes
- `HyVideo15ModelLoader`: Loads the HunyuanVideo-1.5 model.
- `HyVideo15I2VSampler`: Performs the video generation inference.

### Complete Nodes
In addition to the simplified functionality, the complete set includes the following split nodes:
- `HyVideoTextEncode`: Text encoder.
- `HyVideoVisionEncode`: Image encoder.
- Refer to the example workflows for more details.

---

## 🛠 Usage Tips

- Start with the **Simplified Workflow** for initial use to quickly verify results.
- Switch to the **Complete Nodes** for flexible assembly if you need to customize generation logic (e.g., replace encoders, adjust frame sequences).
- Ensure a stable internet connection for auto-download. If download fails, check the path or manually download the models.

---

## ❓ FAQ

**Q: What should I do if auto-download fails?**  
A: Check your network connection, or manually download the models and place them in the corresponding subdirectories under `models/`.

**Q: How do I switch between Simplified and Complete versions?**  
A: Import the corresponding workflow file (e.g., `simplified_I2V_workflow.json` or `complete_I2V_workflow.json`) into ComfyUI. The nodes are grouped by version.

---

## 📄 License

This is an plugin based on the HunyuanVideo-1.5 model. Please comply with the relevant license agreement of the original model.