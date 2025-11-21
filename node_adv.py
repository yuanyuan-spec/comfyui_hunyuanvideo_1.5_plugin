# Copyright (C) 2025 yuanyuan-spec
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import inspect

from typing import Any, Dict, List, Optional, Union

import torch
import loguru
import re
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput
from torch import distributed as dist
import subprocess


from hyvideo.utils.multitask_utils import (
    merge_tensor_by_mask,
)
from hyvideo.commons import (
    PRECISION_TO_TYPE, auto_offload_model, get_gpu_memory,

)
from hyvideo.models.autoencoders import hunyuanvideo_15_vae
from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
from hyvideo.models.transformers.modules.upsample import SRTo720pUpsampler, SRTo1080pUpsampler
from hyvideo.models.vision_encoder import VisionEncoder
from hyvideo.models.text_encoders import TextEncoder, PROMPT_TEMPLATE
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.models.transformers.modules.upsample import SRTo720pUpsampler, SRTo1080pUpsampler

from hyvideo.utils.data_utils import (
    generate_crop_size_list, get_closest_ratio, resize_and_center_crop,
)
from hyvideo.pipelines.pipeline_utils import (retrieve_timesteps, rescale_noise_cfg)
from hyvideo.commons import (SR_PIPELINE_CONFIGS, TRANSFORMER_VERSION_TO_SR_VERSION)
from einops import rearrange

from hyvideo.models.autoencoders import hunyuanvideo_15_vae
from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
from hyvideo.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat
from hyvideo.commons.parallel_states import get_parallel_state

import comfy.model_management as mm
import comfy.utils
import folder_paths

from pathlib import Path
from typing import List




target_size_config = {
    "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
    "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
    "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
    "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
}

dtype_options = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}

def get_immediate_subdirectories(folder_path: str) -> List[str]:
    path = Path(folder_path)
    return sorted([str(item) for item in path.iterdir() if item.is_dir()])

def get_model_dir_path(model_dir):
    all_paths = folder_paths.get_folder_paths(model_dir)
    for path in all_paths:
        if "ComfyUI/models" in path.replace("\\", "/"):
            return path
    return all_paths[0] if all_paths else ""

def tensor_to_pil(comfyui_tensor):
    if comfyui_tensor is None:
        return None
    image_np = comfyui_tensor[0].cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)

def ensure_directories(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)
 

def get_closest_resolution_given_reference_image(reference_image,target_resolution):
    """
    Get closest supported resolution for a reference image.
    Args:
        reference_image: PIL Image or numpy array.
        target_resolution: Target resolution string (e.g., "720p", "1080p").
    Returns:
        tuple[int, int]: (height, width) of closest supported resolution.
    """
    assert reference_image is not None
    if isinstance(reference_image, Image.Image):
        origin_size = reference_image.size
    elif isinstance(reference_image, np.ndarray):
        H, W, C = reference_image.shape
        origin_size = (W, H)
    else:
        raise ValueError(f"Unsupported reference_image type: {type(reference_image)}. Must be PIL Image or numpy array")
    return get_closest_resolution_given_original_size(origin_size, target_resolution)

def get_closest_resolution_given_original_size(origin_size,target_size):
    """
    Get closest supported resolution for given original size and targetresolution.

    Args:
        origin_size: Tuple of (width, height) of original image.
        target_size: Target resolution string (e.g., "720p", "1080p").

    Returns:
        tuple[int, int]: (height, width) of closest supported resolution.
    """
    bucket_hw_base_size = target_size_config[target_size]["bucket_hw_base_size"]
    bucket_hw_bucket_stride = target_size_config[target_size]["bucket_hw_bucket_stride"]
    assert bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960, 1440], \
        f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960, 1440], but got {bucket_hw_base_size}"
    
    crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
    aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
    closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)
    height = closest_size[0]
    width = closest_size[1]
    return height, width

def register_dir():
    comfyui_root = os.path.dirname(folder_paths.__file__)
    model_dir_list = ["diffusion_models"]
    for model_dir in model_dir_list:
        folder_paths.add_model_folder_path(model_dir, os.path.join(comfyui_root, model_dir))
        
register_dir()
    
class HyVideoSRModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": (get_immediate_subdirectories(get_model_dir_path("upscale_models")),),
                "sr_version": (["720p_sr_distilled", "1080p_sr_distilled"], {"default": "720p_sr_distilled"}),
                "transformer_dtype": (["float32","float64","float16","bfloat16","uint8","int8","int16","int32","int64"], {"default": "bfloat16"}),
            },
            "optional": {
            }
        }
    RETURN_TYPES = ("HYVID15TRANSFORMER", "HYVID15UPASAMPLER", )
    RETURN_NAMES = ("transformer", "upsampler",)
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def loadmodel(self, path, sr_version, transformer_dtype="bfloat16"):
        device = mm.get_torch_device()

        transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(os.path.join(path, sr_version), torch_dtype=dtype_options[transformer_dtype]).to(device)
        upsampler_cls = SRTo720pUpsampler if "720p" in sr_version else SRTo1080pUpsampler
        upsampler = upsampler_cls.from_pretrained(os.path.join(path, sr_version)).to(device)
        return transformer, upsampler


class HyVideoTransformerLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING",{"default":""}),
                "resolution": (["480p", "720p"], {"default": "480p"}),
                "task_type": (["t2v", "i2v"], {"default": "i2v"}),
                "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
                "transformer_dtype": (["float32","float64","float16","bfloat16","uint8","int8","int16","int32","int64"], {"default": "bfloat16"}),
            },
            "optional": {
                "attn_mode": (["flash", "flex-block-attn", "ptm_sparse_attn","flash3",], {"default": "flash"}),
            }
        }

    RETURN_TYPES = ("HYVID15TRANSFORMER", "HYVID15TRANSFORMERCONFIG",)
    RETURN_NAMES = ("model", "config", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def loadmodel(self, path, resolution, task_type, transformer_dtype, attn_mode="flash", load_device="main_device"):
        device = mm.get_torch_device() if load_device == "main_device" else mm.unet_offload_device()
        transformer_version = f"{resolution}_{task_type}"

        transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(os.path.join(path, transformer_version), torch_dtype=dtype_options[transformer_dtype]).to(device)
        transformer.set_attn_mode(attn_mode)
        return (transformer, transformer.config)

class HyVideoVaeLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": (get_immediate_subdirectories(get_model_dir_path("vae")),),
                "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("HYVID15VAE",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def loadmodel(self, path, load_device):
        device = mm.get_torch_device() if load_device == "main_device" else mm.unet_offload_device()
        vae_inference_config = self._get_vae_inference_config()

        vae = hunyuanvideo_15_vae.AutoencoderKLConv3D.from_pretrained(path).to(device)
        vae = hunyuanvideo_15_vae.AutoencoderKLConv3D.from_pretrained(
            path,
            torch_dtype=vae_inference_config['dtype']
        ).to(device)
        vae.set_tile_sample_min_size(vae_inference_config['sample_size'], vae_inference_config['tile_overlap_factor'])

        return (vae,)
    
    def _get_vae_inference_config(self,memory_limitation=None):
        if memory_limitation is None:
            memory_limitation = get_gpu_memory()
        GB = 1024 * 1024 * 1024
        if memory_limitation < 23 * GB:
            sample_size = 160
            tile_overlap_factor = 0.2
            dtype = torch.float16
        else:
            sample_size = 256
            tile_overlap_factor = 0.25
            dtype = torch.float32
        return {'sample_size': sample_size, 'tile_overlap_factor': tile_overlap_factor, 'dtype': dtype}

class HyTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": (folder_paths.get_folder_paths("text_encoders"),),
                "text_encoder_type": (["qwen-2.5vl-7b", "llm"], {"default": "llm"}),
                "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("HYVID15TEXTENCODER", "HYVID15TEXTENCODER",)
    RETURN_NAMES = ("text_encoder", "text_encoder_2",)
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def loadmodel(self, path, text_encoder_type, load_device):
        device = mm.get_torch_device() if load_device == "main_device" else mm.unet_offload_device()

        text_encoder = TextEncoder(
            text_encoder_type=text_encoder_type,
            tokenizer_type=text_encoder_type,
            text_encoder_path=os.path.join(path, text_encoder_type),
            max_length=1000,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE['li-dit-encode-image-json'],
            prompt_template_video=PROMPT_TEMPLATE['li-dit-encode-video-json'],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=loguru.logger,
            device=device,

        )
        text_encoder_2 = None

        return text_encoder, text_encoder_2
    
class HyVideoVisionEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": (folder_paths.get_folder_paths("clip_vision"),),
                "vision_encoder_type": (["siglip",], {"default": "siglip"}),
                "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("HYVID15VISIONENCODER",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def loadmodel(self, path, vision_encoder_type, load_device):
        device = mm.get_torch_device() if load_device == "main_device" else mm.unet_offload_device()

        vision_encoder = VisionEncoder(
            vision_encoder_type=vision_encoder_type,
            vision_encoder_precision='fp16',
            vision_encoder_path=os.path.join(path, vision_encoder_type),
            processor_type=None,
            processor_path=None,
            output_key=None,
            logger=loguru.logger,
            device=device,
        )
        return (vision_encoder,)
    
class HyVideoByt5Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": (folder_paths.get_folder_paths("text_encoders"),),
                "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            },
            "optional": {
                "byt5_max_length": ("INT", {"default": 256, "tooltip": "Maximum length for byT5 tokenization."}),
            }
        }

    RETURN_TYPES = ("HYVID15BYT5KWARGS","HYVID15MULTILINGUALPROMPTFORMAT")
    RETURN_NAMES = ("byt5_kwargs","prompt_format" )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def loadmodel(self, path, load_device, byt5_max_length):
        device = mm.get_torch_device() if load_device == "main_device" else mm.unet_offload_device()
        byt5_kwargs, prompt_format = self._load_byt5(path, True, byt5_max_length, device=device)

        return (byt5_kwargs, prompt_format)

    def _load_byt5(self, cached_folder, glyph_byT5_v2, byt5_max_length, device):
        if not glyph_byT5_v2:
            byt5_kwargs = None
            prompt_format = None
            return byt5_kwargs, prompt_format
        try:
            load_from = cached_folder
            glyph_root = os.path.join(load_from, "Glyph-SDXL-v2")
            if not os.path.exists(glyph_root):
                raise RuntimeError(
                    f"Glyph checkpoint not found from '{glyph_root}'. \n"
                    "Please download from https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/files.\n\n"
                    "- Required files:\n"
                    "    Glyph-SDXL-v2\n"
                    "    ├── assets\n"
                    "    │   ├── color_idx.json\n"
                    "    │   └── multilingual_10-lang_idx.json\n"
                    "    └── checkpoints\n"
                    "        └── byt5_model.pt\n"
                )

            byT5_google_path = os.path.join(load_from, "byt5-small")
            if not os.path.exists(byT5_google_path):
                loguru.logger.warning(f"ByT5 google path not found from: {byT5_google_path}. Try downloading from https://huggingface.co/google/byt5-small.")
                byT5_google_path = "google/byt5-small"


            multilingual_prompt_format_color_path = os.path.join(glyph_root, "assets/color_idx.json")
            multilingual_prompt_format_font_path = os.path.join(glyph_root, "assets/multilingual_10-lang_idx.json")

            byt5_args = dict(
                byT5_google_path=byT5_google_path,
                byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
                multilingual_prompt_format_color_path=multilingual_prompt_format_color_path,
                multilingual_prompt_format_font_path=multilingual_prompt_format_font_path,
                byt5_max_length=byt5_max_length
            )

            byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=device)
            prompt_format = MultilingualPromptFormat(
                font_path=multilingual_prompt_format_font_path,
                color_path=multilingual_prompt_format_color_path
            )
            return byt5_kwargs, prompt_format
        except Exception as e:
            raise RuntimeError("Error loading byT5 glyph processor") from e

class HyVideoCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A close-up shot captures a scene on a polished, light-colored granite kitchen counter, illuminated by soft natural light from an unseen window. Initially, the frame focuses on a tall, clear glass filled with golden, translucent apple juice standing next to a single, shiny red apple with a green leaf still attached to its stem. The camera moves horizontally to the right. As the shot progresses, a white ceramic plate smoothly enters the frame, revealing a fresh arrangement of about seven or eight more apples, a mix of vibrant reds and greens, piled neatly upon it. A shallow depth of field keeps the focus sharply on the fruit and glass, while the kitchen backsplash in the background remains softly blurred. The scene is in a realistic style.", "multiline": True} ),
                "negative_prompt": ("STRING", {"default": "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion", "multiline": True, "tooltip": "Negative prompt(s) that describe what should NOT be shown in the generated video."} ),
                "prompt_rewrite": ("BOOLEAN", {"default": False, "tooltip": "Whether to rewrite the prompt."}),
                "flow_shift": ("FLOAT",{"default": None, "tooltip": "When the resolution is 480p, the recommended shift value is 5, and when the resolution is 720p, the recommended shift value is 7. If you do not set this value, it will be automatically configured according to the recommendations of this rule."}),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {"default": 6.0, "tooltip": "Scale to encourage the model to better follow the prompt. `guidance_scale > 1` enables classifier-free guidance."}),
                "num_videos_per_prompt": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Number of videos to generate per prompt."} ),
                "video_length": ("INT", {"default": 121, "min": 1, "max": 200, "step": 1, "tooltip": "Number of frames to generate."} ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1} ),
                "transformer_config": ("HYVID15TRANSFORMERCONFIG", ),
                "task_type": (["t2v", "i2v"], {"default": "i2v"}),
                "sr_transformer_config": ("HYVID15TRANSFORMERCONFIG", {"default": None}),
                "prompt_rewrite":("BOOLEAN", {"default": False, "tooltip": "Rewrite prompt."}),
                "download_model":("BOOLEAN", {"default": False, "tooltip": "If Download model."}),
                "hf_token": ("STRING",{"default":""}),
                "reference_image": ("IMAGE", {"default": None, "tooltip": "Reference image."}),
            }
        }

    RETURN_TYPES = ("HYVID15CFG","HYVID15CFG",)
    RETURN_NAMES = ("hyvid_cfg","hyvid_sr_cfg",)
    FUNCTION = "config"
    CATEGORY = "HunyuanVideoWrapper1.5"
    DESCRIPTION = "To use CFG with HunyuanVideo"

    def config(self, prompt, negative_prompt, guidance_scale, num_videos_per_prompt, video_length, seed, transformer_config, flow_shift=None, prompt_rewrite=False, reference_image=None,download_model=False,hf_token="",task_type="i2v",sr_transformer_config=None): 
        
        if flow_shift is None or flow_shift == 0:
            if isinstance(transformer_config.ideal_resolution, str) and transformer_config.ideal_resolution == "480p":
                flow_shift = 5.0
            else:
                flow_shift = 7.0
    
        device = mm.get_torch_device()
        generator = torch.Generator(device=device).manual_seed(seed)
        self.prompt = prompt
        self.reference_image = reference_image
        self.task_type = task_type

        if download_model:
            text_encoderd_dir = self.get_model_dir_path("text_encoders")
            vision_encoderd_dir = self.get_model_dir_path("clip_vision")
            upscale_dir = os.path.join(self.get_model_dir_path("upscale_models"), "hyvideo15")
            vae_dir = os.path.join(self.get_model_dir_path("vae"), "hyvideo15")
            diffusion_dir = os.path.join(self.get_model_dir_path("diffusion_models"), "hyvideo15")
            
            temp_dir = comfy.utils.get_temp_directory()
            hunyuan_dir = os.path.join(temp_dir, "hyvideo15")
            
            ensure_directories([text_encoderd_dir,vision_encoderd_dir,upscale_dir,vae_dir,diffusion_dir,hunyuan_dir])
            
            if not os.path.exists(hunyuan_dir):
                cmd_list = [
                    f"hf download tencent/HunyuanVideo-1.5 --local-dir {hunyuan_dir}",
                    f"cp {hunyuan_dir}/upsampler/* {upscale_dir} -r -f",
                    f"cp {hunyuan_dir}/vae/* {vae_dir} -r -f",
                    f"cp {hunyuan_dir}/transformer/* {diffusion_dir} -r -f",
                ]
                for cmd in cmd_list:
                    self._cmd(cmd)
                
            if not os.path.exists(f"{text_encoderd_dir}/llm"):
                self._cmd(f"hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir {text_encoderd_dir}/llm")
                
            if not os.path.exists(f"{text_encoderd_dir}/byt5-small"):
                self._cmd(f"hf download google/byt5-small --local-dir {text_encoderd_dir}/byt5-small")
                
            if not os.path.exists(f"{text_encoderd_dir}/Glyph-SDXL-v2"):
                self._cmd(f"modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir {text_encoderd_dir}/Glyph-SDXL-v2")
                
            if not os.path.exists(f"{vision_encoderd_dir}/siglip"):
                self._cmd(f"hf download black-forest-labs/FLUX.1-Redux-dev --local-dir {vision_encoderd_dir}/siglip --token {hf_token}")
            
            

                


        
        # Rewrite prompt with QwenClient
        if prompt_rewrite:
            from hyvideo.utils.rewrite.rewrite_utils import run_prompt_rewrite

            if not dist.is_initialized() or get_parallel_state().sp_rank == 0:
                prompt = run_prompt_rewrite(prompt, reference_image, task_type)

            if dist.is_initialized() and get_parallel_state().sp_enabled:
                obj_list = [prompt]
                dist.broadcast_object_list(obj_list, group_src=0, group=get_parallel_state().sp_group)
                prompt = obj_list[0]


        self.prompt = prompt
        self.reference_image = reference_image
        self.task_type = transformer_config.ideal_task
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1
        if prompt_rewrite:
            prompt = self._prompt_rewrite(prompt)

        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=True,
            solver="euler",
        )

        cfg_dict = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_videos_per_prompt": num_videos_per_prompt,
            "video_length": video_length,
            "seed": seed,
            "task_type": task_type,
            "generator": generator,
            "scheduler": scheduler,
            "transformer_config": transformer_config,
            "sr_transformer_config": sr_transformer_config,
            "batch_size": batch_size,
            "flow_shift": flow_shift,
        }
        
        return (cfg_dict,)
    
    def _prompt_rewrite(self, prompt):
        from hyvideo.utils.rewrite.rewrite_utils import run_prompt_rewrite

        if not dist.is_initialized() or get_parallel_state().sp_rank == 0:
            try:
                prompt = run_prompt_rewrite(prompt, self.reference_image, self.task_type)
            except Exception as e:
                loguru.logger.warning(f"Failed to rewrite prompt: {e}")
                prompt = prompt
            
        if dist.is_initialized() and get_parallel_state().sp_enabled:
            obj_list = [prompt]
            # not use group_src to support old PyTorch
            group_src_rank = dist.get_global_rank(get_parallel_state().sp_group, 0)
            dist.broadcast_object_list(obj_list, src=group_src_rank, group=get_parallel_state().sp_group)
            prompt = obj_list[0]

        return prompt

    def _prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    def _cmd(self,cmd):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True,check=True)
        
    def get_model_dir_path(model_dir):
        all_paths = folder_paths.get_folder_paths(model_dir)
        for path in all_paths:
            if "ComfyUI/models" in path.replace("\\", "/"):
                return path

        return all_paths[0] if all_paths else ""


class HyVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text_encoder": ("HYVID15TEXTENCODER",),
            "text_encoder_2": ("HYVID15TEXTENCODER",),
            "hyvid_cfg": ("HYVID15CFG", ),
            },
            "optional": {
                "enable_offloading": ("BOOLEAN", {"default": True}),
                "clip_skip": ("INT", {"default": 0, "tooltip": "Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that the output of the pre-final layer will be used for computing the prompt embeddings."}),
            }
        }

    RETURN_TYPES = ("HYVIDEMBEDS", )
    RETURN_NAMES = ("hyvid_embeds",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def process(self, text_encoder, text_encoder_2, hyvid_cfg, enable_offloading=True, clip_skip=0):
        negative_prompt = hyvid_cfg["negative_prompt"]
        do_classifier_free_guidance = hyvid_cfg["guidance_scale"] > 1
        device = mm.text_encoder_offload_device() if enable_offloading else mm.text_encoder_device()
        
        self.text_len = text_encoder.max_length
        self.text_encoder = text_encoder
        

        with auto_offload_model(text_encoder, device, enabled=enable_offloading):
            (
                prompt_embeds,
                negative_prompt_embeds,
                prompt_mask,
                negative_prompt_mask,
            ) = self._encode_prompt(
                hyvid_cfg["prompt"],
                device,
                hyvid_cfg["num_videos_per_prompt"],
                do_classifier_free_guidance,
                negative_prompt,
                clip_skip=None if clip_skip == 0 else clip_skip,
                data_type="video",
                text_encoder=text_encoder,
            )

        # Encode prompts with second encoder if available
        if text_encoder_2 is not None:
            with auto_offload_model(text_encoder_2, device, enabled=enable_offloading):
                (
                    prompt_embeds_2,
                    negative_prompt_embeds_2,
                    prompt_mask_2,
                    negative_prompt_mask_2,
                ) = self._encode_prompt(
                    hyvid_cfg["prompt"],
                    device,
                    hyvid_cfg["num_videos_per_prompt"],
                    do_classifier_free_guidance,
                    negative_prompt,
                    clip_skip=None if clip_skip == 0 else clip_skip,
                    text_encoder=text_encoder_2,
                    data_type="video",
                )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        hyvid_embeds = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "prompt_mask": prompt_mask,
            "negative_prompt_mask": negative_prompt_mask,
            "prompt_embeds_2": prompt_embeds_2,
            "negative_prompt_embeds_2": negative_prompt_embeds_2,
            "prompt_mask_2": prompt_mask_2,
            "negative_prompt_mask_2": negative_prompt_mask_2,
        }
        return (hyvid_embeds, )
    
    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
                Text encoder to use. If None, uses the pipeline's default text encoder.
            data_type (`str`, *optional*):
                Type of data being encoded. Defaults to "image".
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:

            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type, max_length=self.text_len)
            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, device=device
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                    device=device,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type, max_length=self.text_len)

            negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type, is_uncond=True)
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(1, num_videos_per_prompt)
                negative_attention_mask = negative_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, -1)
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

class HyVideoGlyphByT5:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "byt5_kwargs": ("HYVID15BYT5KWARGS", ),
                "prompt_format" : ("HYVID15MULTILINGUALPROMPTFORMAT", ),
            },
            "optional": {
                "enable_offloading": ("BOOLEAN", {"default": True}),
                "hyvid_cfg": ("HYVID15CFG", ),
            }
        }

    RETURN_TYPES = ("HYVID15EXTRAKWARGS", )
    RETURN_NAMES = ("extra_kwargs",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def process(self, byt5_kwargs, hyvid_cfg, prompt_format, enable_offloading=True):
        extra_kwargs = {}
        device = mm.text_encoder_offload_device() if enable_offloading else mm.text_encoder_device()

        self.byt5_kwargs = byt5_kwargs
        self.prompt_format = prompt_format
        self.hyvid_cfg = hyvid_cfg
        with auto_offload_model(byt5_kwargs["byt5_model"], device, enabled=enable_offloading):
            extra_kwargs = self._prepare_byt5_embeddings(hyvid_cfg["prompt"], device)
        return (extra_kwargs,)

    def _prepare_byt5_embeddings(self, prompts, device):
        """
        Prepare byT5 embeddings for both positive and negative prompts.

        Args:
            prompts: List of prompt strings or single prompt string.
            device: Target device for tensors.

        Returns:
            dict: Dictionary containing:
                - "byt5_text_states": Combined embeddings tensor.
                - "byt5_text_mask": Combined attention mask tensor.
                Returns empty dict if glyph_byT5_v2 is disabled.
        """
        if isinstance(prompts, str):
            prompt_list = [prompts]
        elif isinstance(prompts, list):
            prompt_list = prompts
        else:
            raise ValueError("prompts must be str or list of str")

        positive_embeddings = []
        positive_masks = []
        negative_embeddings = []
        negative_masks = []

        for prompt in prompt_list:
            pos_emb, pos_mask = self._process_single_byt5_prompt(prompt, device)
            positive_embeddings.append(pos_emb)
            positive_masks.append(pos_mask)

            if self.hyvid_cfg["guidance_scale"] > 1:
                neg_emb, neg_mask = self._process_single_byt5_prompt("", device)
                negative_embeddings.append(neg_emb)
                negative_masks.append(neg_mask)

        byt5_positive = torch.cat(positive_embeddings, dim=0)
        byt5_positive_mask = torch.cat(positive_masks, dim=0)
        
        if self.hyvid_cfg["guidance_scale"] > 1:
            byt5_negative = torch.cat(negative_embeddings, dim=0)
            byt5_negative_mask = torch.cat(negative_masks, dim=0)
            
            byt5_embeddings = torch.cat([byt5_negative, byt5_positive], dim=0)
            byt5_masks = torch.cat([byt5_negative_mask, byt5_positive_mask], dim=0)
        else:
            byt5_embeddings = byt5_positive
            byt5_masks = byt5_positive_mask

        return {
            "byt5_text_states": byt5_embeddings,
            "byt5_text_mask": byt5_masks
        }

    def _process_single_byt5_prompt(self, prompt_text, device):
        """
        Process a single prompt for byT5 encoding.

        Args:
            prompt_text: The prompt text to process.
            device: Target device for tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - byt5_embeddings: Encoded embeddings tensor.
                - byt5_mask: Attention mask tensor.
        """
        byt5_embeddings = torch.zeros((1, self.byt5_kwargs["byt5_max_length"], 1472), device=device)
        byt5_mask = torch.zeros((1, self.byt5_kwargs["byt5_max_length"]), device=device, dtype=torch.int64)
        
        glyph_texts = self._extract_glyph_texts(prompt_text)
        
        if len(glyph_texts) > 0:
            text_styles = [{'color': None, 'font-family': None} for _ in range(len(glyph_texts))]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)
            
            text_ids, text_mask = self._get_byt5_text_tokens(
                self.byt5_kwargs["byt5_tokenizer"], self.byt5_kwargs["byt5_max_length"], formatted_text
            )
            text_ids = text_ids.to(device=device)
            text_mask = text_mask.to(device=device)
            
            byt5_outputs = self.byt5_kwargs["byt5_model"](text_ids, attention_mask=text_mask.float())
            byt5_embeddings = byt5_outputs[0]
            byt5_mask = text_mask
            
        return byt5_embeddings, byt5_mask
    
    def _extract_glyph_texts(self, prompt):
        """
        Extract glyph texts from prompt using regex pattern.

        Args:
            prompt: Input prompt string containing quoted text.

        Returns:
            List[str]: List of extracted glyph texts (deduplicated if multiple).
        """
        pattern = r'\"(.*?)\"|“(.*?)”'
        matches = re.findall(pattern, prompt)
        result = [match[0] or match[1] for match in matches]
        result = list(dict.fromkeys(result)) if len(result) > 1 else result
        return result
    
    @staticmethod
    def _get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
        """
        Tokenize text prompt for byT5 model.

        Args:
            byt5_tokenizer: The byT5 tokenizer.
            byt5_max_length: Maximum sequence length for tokenization.
            text_prompt: Text prompt string to tokenize.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - input_ids: Tokenized input IDs.
                - attention_mask: Attention mask tensor.
        """
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return byt5_text_inputs.input_ids, byt5_text_inputs.attention_mask

class HyVideoVisionEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vision_encoder": ("HYVID15VISIONENCODER", ),
                "hyvid_cfg": ("HYVID15CFG", ),
                "latents_dict": ("HYVID15LATENTSDICT", ),
                "target_dtype": (["float32","float64","float16","bfloat16","uint8","int8","int16","int32","int64"], {"default": "bfloat16"}),
            },
            "optional": {
                "enable_offloading": ("BOOLEAN", {"default": True}),
                "reference_image": ("IMAGE", {"default": None}),
                "vision_num_semantic_tokens": ("INT", {"default": 729}),
                "vision_states_dim": ("INT", {"default": 1152}),
            }
        }
    RETURN_TYPES = ("HYVID15VISIONSTATES", )
    RETURN_NAMES = ("vision_states",)
    FUNCTION = "encode"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def encode(self, vision_encoder, hyvid_cfg, latents_dict, enable_offloading=True, reference_image=None, vision_num_semantic_tokens=729, vision_states_dim=1152, target_dtype="bfloat16"):
        self.vision_encoder = vision_encoder
        self.do_classifier_free_guidance = hyvid_cfg["guidance_scale"] > 1
        self.vision_num_semantic_tokens = vision_num_semantic_tokens
        self.vision_states_dim = vision_states_dim
        self.target_dtype = dtype_options[target_dtype]
        if reference_image is not None:
            reference_image = tensor_to_pil(reference_image)
        semantic_images_np = None if reference_image is None else np.array(reference_image)
        device = mm.unet_offload_device() if enable_offloading else mm.get_torch_device()
        
        target_resolution = hyvid_cfg["transformer_config"].ideal_resolution
        with auto_offload_model(vision_encoder, device, enabled=enable_offloading):
            vision_states = self._prepare_vision_states(
                semantic_images_np, target_resolution, latents_dict["latents"], device
            )
        return (vision_states,)

    def _prepare_vision_states(self, reference_image, target_resolution, latents, device):
        """
        Prepare vision states for multitask training.

        Args:
            reference_image: Reference image for i2v tasks (None for t2v tasks).
            target_resolution: Target size for i2v tasks.
            latents: Latent tensors.
            device: Target device.

        Returns:
            torch.Tensor or None: Vision states tensor or None if vision encoder is unavailable.
        """
        
        if reference_image is None:
            vision_states = torch.zeros(latents.shape[0], self.vision_num_semantic_tokens, self.vision_states_dim).to(latents.device)
        else:
            reference_image = np.array(reference_image) if isinstance(reference_image, Image.Image) else reference_image
            if len(reference_image.shape) == 4:
                reference_image = reference_image[0]

            height, width = get_closest_resolution_given_reference_image(reference_image, target_resolution)

            # Encode reference image to vision states
            if self.vision_encoder is not None:
                input_image_np = resize_and_center_crop(reference_image, target_width=width, target_height=height)
                vision_states = self.vision_encoder.encode_images(input_image_np)
                vision_states = vision_states.last_hidden_state.to(device=device, dtype=self.target_dtype)
            else:
                vision_states = None
        
        # Repeat image features for batch size if needed (for classifier-free guidance)
        if self.do_classifier_free_guidance and vision_states is not None:
            vision_states = vision_states.repeat(2, 1, 1)
        
        return vision_states

    
class HyVideoVaeEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("HYVID15VAE", ),
                "latents_dict": ("HYVID15LATENTSDICT", ),
                "height": ("INT", {"default": 768}),
                "width": ("INT", {"default": 512}),
                "hyvid_cfg": ("HYVID15CFG", )
            },
            "optional": {
                "enable_offloading": ("BOOLEAN", {"default": True}),
                "reference_image": ("IMAGE", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("HYVID15VAECONCAT", )
    RETURN_NAMES = ("vae_concat", )
    FUNCTION = "encode"
    CATEGORY = "HunyuanVideoWrapper1.5"
    def encode(self, vae, latents_dict, height, width, hyvid_cfg, enable_offloading=True, reference_image=None):
        self.vae = vae
        device = mm.vae_offload_device() if enable_offloading else mm.vae_device()
        multitask_mask = self._get_task_mask(hyvid_cfg["task_type"], latents_dict["latent_target_length"])

        if reference_image is not None:
            reference_image = tensor_to_pil(reference_image)
        with auto_offload_model(vae, device, enabled=enable_offloading): 
            image_cond = self._get_image_condition_latents(device, reference_image, height, width)

        cond_latents = self._prepare_cond_latents(
            hyvid_cfg["task_type"], image_cond, latents_dict["latents"], multitask_mask
        )

        return (cond_latents, )
    
    def _prepare_cond_latents(self, task_type, cond_latents, latents, multitask_mask):
        """
        Prepare conditional latents and mask for multitask training.

        Args:
            task_type: Type of task ("i2v" or "t2v").
            cond_latents: Conditional latents tensor.
            latents: Main latents tensor.
            multitask_mask: Multitask mask tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - latents_concat: Concatenated conditional latents.
                - mask_concat: Concatenated mask tensor.
        """
        latents_concat = None
        mask_concat = None
        
        if cond_latents is not None and task_type == 'i2v':
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3], latents.shape[4]).to(latents.device)
        
        mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(device=latents.device)

        cond_latents = torch.concat([latents_concat, mask_concat], dim=1)
        
        return cond_latents

    def _get_image_condition_latents(self, device, reference_image, height, width):
        if reference_image is None:
            cond_latents = None
        else:
            origin_size = reference_image.size
            
            target_height, target_width = height, width
            original_width, original_height = origin_size
            
            scale_factor = max(target_width / original_width, target_height / original_height)
            resize_width = int(round(original_width * scale_factor))
            resize_height = int(round(original_height * scale_factor))
            
            ref_image_transform = transforms.Compose([
                transforms.Resize((resize_height, resize_width),
                                  interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop((target_height, target_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            
            ref_images_pixel_values = ref_image_transform(reference_image).unsqueeze(0).unsqueeze(2).to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                cond_latents = self.vae.encode(ref_images_pixel_values).latent_dist.mode()
                cond_latents.mul_(self.vae.config.scaling_factor)
                    
        return cond_latents

    def _get_task_mask(self, task_type, latent_target_length):
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported !")
        return mask


class HyVideoLatentsPrepare:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hyvid_cfg": ("HYVID15CFG", ),
                "vae": ("HYVID15VAE", ),
                "aspect_ratio": ("STRING", {"default": "16:9"}),
                "target_dtype": (["float32","float64","float16","bfloat16","uint8","int8","int16","int32","int64"], {"default": "bfloat16"}),
            },
            "optional": {
                "latents": ("LATENT", ),
                "reference_image": ("IMAGE", {"default": None}),
            }
        }
    RETURN_TYPES = ("HYVID15LATENTSDICT", "INT", "INT", "INT")
    RETURN_NAMES = ("latents_dict", "height", "width", "n_tokens")
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def process(self, hyvid_cfg,vae, reference_image=None, aspect_ratio="16:9", latents=None, target_dtype="bfloat16"):
        self.vae = vae
        device = mm.get_torch_device()
        if reference_image is not None:
            reference_image = tensor_to_pil(reference_image)
        self.scheduler = hyvid_cfg["scheduler"]
        target_resolution = hyvid_cfg["transformer_config"].ideal_resolution
        if reference_image is not None:
            height, width = get_closest_resolution_given_reference_image(reference_image, target_resolution)
        else:
            if target_resolution is not None:
                if ":" not in aspect_ratio:
                    raise ValueError("aspect_ratio must be separated by a colon")
                width, height = aspect_ratio.split(":")
                # check if width and height are integers
                if not width.isdigit() or not height.isdigit() or int(width) <= 0 or int(height) <= 0:
                    raise ValueError("width and height must be positive integers and separated by a colon in aspect_ratio")
                width = int(width)
                height = int(height)
                height, width = get_closest_resolution_given_original_size((width, height), target_resolution)

        latent_target_length, latent_height, latent_width = self._get_latent_size(hyvid_cfg["video_length"], height, width)
        n_tokens = latent_target_length * latent_height * latent_width

        num_channels_latents = hyvid_cfg["transformer_config"].in_channels
        latents = self._prepare_latents(
            hyvid_cfg["batch_size"] * hyvid_cfg["num_videos_per_prompt"],
            num_channels_latents,
            latent_height,
            latent_width,
            latent_target_length,
            dtype_options[target_dtype],
            device,
            hyvid_cfg["generator"],
        )

        latents_dict = {
            "latents": latents,
            "latent_target_length": latent_target_length,
        }
        
        return (latents_dict, height, width, n_tokens)
    
    #TODO: read from vae config
    @property
    def _vae_spatial_compression_ratio(self):
        if hasattr(self.vae.config, "ffactor_spatial"):
            return self.vae.config.ffactor_spatial
        else:
            return 16

    @property
    def _vae_temporal_compression_ratio(self):
        if hasattr(self.vae.config, "ffactor_temporal"):
            return self.vae.config.ffactor_temporal
        else:
            return 4

    def _get_latent_size(self, video_length, height, width):
        spatial_compression_ratio = self._vae_spatial_compression_ratio
        temporal_compression_ratio = self._vae_temporal_compression_ratio
        video_length = (video_length - 1) // temporal_compression_ratio + 1
        height, width = height // spatial_compression_ratio, width // spatial_compression_ratio

        assert height > 0 and width > 0 and video_length > 0, f"height: {height}, width: {width}, video_length: {video_length}"
        
        return video_length, height, width
    
    def _prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        latent_height,
        latent_width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """
        Prepare latents for video generation.

        Args:
            batch_size: Batch size for generation.
            num_channels_latents: Number of channels in latent space.
            latent_height: Height of latent tensors.
            latent_width: Width of latent tensors.
            video_length: Number of frames in the video.
            dtype: Data type for latents.
            device: Target device for latents.
            generator: Random number generator.
            latents: Pre-computed latents. If None, random latents are generated.

        Returns:
            torch.Tensor: Prepared latents tensor.
        """

        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            latent_height,
            latent_width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents
    

class HyVideoTransformer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hyvid_cfg": ("HYVID15CFG", ),
                "n_tokens": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 1000, "step": 1}),
                "transformer": ("HYVID15TRANSFORMER", ),
                "vae_concat": ("HYVID15VAECONCAT", ),
                "hyvid_embeds": ("HYVIDEMBEDS", ),
                "vision_states": ("HYVID15VISIONSTATES", ),
                "extra_kwargs": ("HYVID15EXTRAKWARGS", ),
                "latents_dict": ("HYVID15LATENTSDICT", ),
                "target_dtype": (["float32","float64","float16","bfloat16","uint8","int8","int16","int32","int64"], {"default": "bfloat16"}),
            },
            "optional": {
                "enable_offloading" : ("BOOLEAN", {"default": True}),
                "embedded_guidance_scale": ("FLOAT", {"default": None, "tooltip": "Additional control guidance scale, if supported"}), #self.config.embedded_guidance_scale
                "guidance_rescale" : ("FLOAT", {"default": 0.0}),
                "autocast_enabled": ("BOOLEAN", {"default": True}),
                "eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01} ),
            }
        }
    RETURN_TYPES = ("HYVID15TRANSFORMERLATENT", )
    RETURN_NAMES = ("transformer_latent",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def process(self, hyvid_cfg, n_tokens, steps, transformer, vae_concat, hyvid_embeds, vision_states, extra_kwargs, latents_dict, target_dtype, enable_offloading=True, embedded_guidance_scale=None, autocast_enabled=True, guidance_rescale=0.0, eta=0.0):
        # print("vae_concat: ", vae_concat)
        # print("extra_kwargs: ", extra_kwargs)
        # print("latens_dict: ", latens_dict)
        # print("hyvid_embeds: ", hyvid_embeds)
        # print("transformer: ", transformer)
        # print("vision_states: ", vision_states)
        # print("hyvid_cfg: ", hyvid_cfg)
        # print("n_tokens: ", n_tokens)
        extra_step_kwargs = self._prepare_extra_func_kwargs(
            hyvid_cfg["scheduler"].step, {"generator": hyvid_cfg["generator"], "eta": eta},
        )
        extra_set_timesteps_kwargs = self._prepare_extra_func_kwargs(
            hyvid_cfg["scheduler"].set_timesteps, {"n_tokens": n_tokens}
        )
        device = mm.get_torch_device() if not enable_offloading else mm.unet_offload_device()
        timesteps, num_inference_steps = retrieve_timesteps(
            hyvid_cfg["scheduler"],
            steps,
            device,
            **extra_set_timesteps_kwargs,
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * hyvid_cfg["scheduler"].order
        latents = latents_dict["latents"]
        cond_latents = vae_concat
        
        with  auto_offload_model(transformer, device, enabled=enable_offloading):
            progress_bar = self._progress_bar(total=num_inference_steps)
            self.do_classifier_free_guidance = hyvid_cfg["guidance_scale"] > 1
            for i, t in enumerate(timesteps):
                latents_concat = torch.concat([latents, cond_latents], dim=1)

                latent_model_input = torch.cat([latents_concat] * 2) if self.do_classifier_free_guidance else latents_concat

                latent_model_input = hyvid_cfg["scheduler"].scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])
                if hyvid_cfg["transformer_config"].use_meanflow:
                    if i == len(timesteps) - 1:
                        timesteps_r = torch.tensor([0.0], device=device)
                    else:
                        timesteps_r = timesteps[i + 1]
                    timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
                else:
                    timesteps_r = None

                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(dtype_options[target_dtype])
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                with torch.autocast(device_type="cuda", dtype=dtype_options[target_dtype], enabled=autocast_enabled):
                    output = transformer(
                        latent_model_input,
                        t_expand,
                        hyvid_embeds["prompt_embeds"],
                        hyvid_embeds["prompt_embeds_2"],
                        hyvid_embeds["prompt_mask"],
                        timestep_r=timesteps_r,
                        vision_states=vision_states,
                        mask_type=hyvid_cfg["task_type"],
                        guidance=guidance_expand,
                        return_dict=False,
                        extra_kwargs=extra_kwargs,
                    )
                    noise_pred = output[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + hyvid_cfg["guidance_scale"] * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = hyvid_cfg["scheduler"].step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % hyvid_cfg["scheduler"].order == 0):
                    if progress_bar is not None:
                        progress_bar.update(1)

        return (latents,)
    
    def _progress_bar(self, total, desc="Processing"):
        """使用ComfyUI进度条"""
        pbar = comfy.utils.ProgressBar(total)
        return pbar

    def _prepare_extra_func_kwargs(self, func, kwargs):
        """
        Prepare extra keyword arguments for scheduler functions.

        Filters kwargs to only include parameters that the function accepts.
        This is useful since not all schedulers have the same signature.
        """
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs


class HyVideoSR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transformer": ("HYVID15TRANSFORMER", ),
                "upsampler": ("HYVID15UPASAMPLER", ),
                "text_encoder": ("HYVID15TEXTENCODER", ),
                "text_encoder_2": ("HYVID15TEXTENCODER", ),
                "vae": ("HYVID15VAE", ),
                "steps": ("INT", {"default": 50, "min": 1, "max": 1000, "step": 1}),
                "hyvid_cfg": ("HYVID15CFG", ),
                "output_type": (["pt", "latent"], {"default": "pt" }),
                "latents": ("HYVID15TRANSFORMERLATENT", ),
                "reference_image": ("IMAGE", {"default": None}),
                "prompt_format" : ("HYVID15MULTILINGUALPROMPTFORMAT", ),
                "vision_encoder": ("HYVID15VISIONENCODER", ),
            },
            "optional": {
                "enable_offloading" : ("BOOLEAN", {"default": True}),
                "transformer_version": ("STRING", {"default": "480p_t2v", "options": ["480p_t2v", "720p_t2v", "480p_i2v", "720p_i2v"]}),
                "byt5_kwargs": ("HYVID15BYT5KWARGS", {"default": None}),
            }
        }
    RETURN_TYPES = ("HYVID15SROUT", )
    RETURN_NAMES = ("sr_out",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper1.5"
    def process(self, steps, hyvid_cfg, output_type, latents, transformer, upsampler,  text_encoder, text_encoder_2, vae, prompt_format, vision_encoder, enable_offloading=True, transformer_version="480p_t2v", byt5_kwargs=None,reference_image=None):
        self.enable_offloading = enable_offloading
        self.transformer = transformer
        self.upsampler = upsampler
        self.hyvid_cfg = hyvid_cfg
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.byt5_kwargs = byt5_kwargs
        self.prompt_format = prompt_format
        self.vision_encoder = vision_encoder
        if reference_image is not None:
            reference_image = tensor_to_pil(reference_image)

        sr_version = TRANSFORMER_VERSION_TO_SR_VERSION[transformer_version]
        sr_pipeline = self._create_sr_pipeline(sr_version)
        sr_out = sr_pipeline(
            prompt=hyvid_cfg["prompt"],
            num_inference_steps=steps,
            video_length=hyvid_cfg["video_length"],
            negative_prompt="",
            num_videos_per_prompt=hyvid_cfg["num_videos_per_prompt"],
            seed=hyvid_cfg["seed"],
            output_type=output_type,
            lq_latents=latents,
            reference_image=reference_image,
        )
        return (sr_out,)
    
    def _create_sr_pipeline(self, sr_version):
        from hyvideo.pipelines.hunyuan_video_sr_pipeline import HunyuanVideo_1_5_SR_Pipeline

        return HunyuanVideo_1_5_SR_Pipeline(
            vae=self.vae,
            transformer=self.transformer,
            text_encoder=self.text_encoder,
            scheduler=self.hyvid_cfg["scheduler"],
            upsampler=self.upsampler,
            text_encoder_2=self.text_encoder_2,
            progress_bar_config=None,
            glyph_byT5_v2=True if self.byt5_kwargs is not None else False,
            byt5_model=self.byt5_kwargs["byt5_model"],
            byt5_tokenizer=self.byt5_kwargs["byt5_tokenizer"],
            byt5_max_length=self.byt5_kwargs["byt5_max_length"],
            prompt_format=self.prompt_format,
            execution_device='cuda',
            vision_encoder=self.vision_encoder,
            enable_offloading=self.enable_offloading,
            **SR_PIPELINE_CONFIGS[sr_version],
        )

class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    sr_videos: Union[torch.Tensor, np.ndarray]

class HyVideoVaeDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("HYVID15TRANSFORMERLATENT", ),
                "output_type": (["pt", "latent"], {"default": "pt"}),
                "vae": ("HYVID15VAE", ),
                "hyvid_cfg": ("HYVID15CFG", ),
                "vae_dtype": (["float32","float64","float16","bfloat16","uint8","int8","int16","int32","int64"], {"default": "float16"}),
            },
            "optional": {
                "enable_offloading" : ("BOOLEAN", {"default": True}),
                "sr_out": ("HYVID15SROUT", ),
                "vae_autocast_enabled": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper1.5"
    def process(self, latents, output_type, vae, hyvid_cfg, enable_offloading=True, vae_dtype="float16", sr_out=None, vae_autocast_enabled=True):
        device = mm.vae_offload_device() if enable_offloading else mm.vae_device()
        self.vae = vae
        self.vae_dtype = dtype_options[vae_dtype]
        self.vae_autocast_enabled = vae_autocast_enabled
        if output_type == "latent":
            video_frames = latents
        else:
            if len(latents.shape) == 4:
                latents = latents.unsqueeze(2)
            elif len(latents.shape) != 5:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            else:
                latents = latents / self.vae.config.scaling_factor

            if hasattr(self.vae, 'enable_tile_parallelism'):
                self.vae.enable_tile_parallelism()

            with torch.autocast(device_type="cuda", dtype=self.vae_dtype, enabled=self.vae_autocast_enabled), auto_offload_model(self.vae, device, enabled=enable_offloading):
                self.vae.enable_tiling()
                video_frames = self.vae.decode(latents, return_dict=False, generator=hyvid_cfg["generator"])[0]
                self.vae.disable_tiling()

            if video_frames is not None:
                video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()

        if sr_out is not None:
            sr_video_frames = sr_out.videos
            result = HunyuanVideoPipelineOutput(videos=video_frames, sr_videos=sr_video_frames)
        else:
            result = HunyuanVideoPipelineOutput(videos=video_frames)
        
        if output_type == "latent":
            if sr_out is not None:
                return result.sr_videos
            else:
                return result.videos
        
        if sr_out is not None:
            video_tensor = result.sr_videos.permute(0, 2, 3, 4, 1)
        else:
            video_tensor = result.videos.permute(0, 2, 3, 4, 1)
        
        if hyvid_cfg["num_videos_per_prompt"] == 1:
            return (video_tensor[0],)
        else:
            return (video_tensor,)


NODE_CLASS_MAPPINGS = {
    "HyVideo15TransformerLoader": HyVideoTransformerLoader,
    "HyVideo15VaeLoader": HyVideoVaeLoader,
    "HyVideo15TextEncoderLoader": HyTextEncoderLoader,
    "HyVideo15VisionEncoderLoader": HyVideoVisionEncoderLoader,
    "HyVideo15Byt5Loader": HyVideoByt5Loader,
    "HyVideo15CFG": HyVideoCFG,
    "HyVideo15TextEncode": HyVideoTextEncode,
    "HyVideo15GlyphByT5": HyVideoGlyphByT5,
    "HyVideo15VisionEncode": HyVideoVisionEncode,
    "HyVideo15VaeEncode": HyVideoVaeEncode,
    "HyVideo15Transformer": HyVideoTransformer,
    "HyVideo15LatentsPrepare": HyVideoLatentsPrepare,
    "HyVideo15SR": HyVideoSR,
    "HyVideo15SRModelLoader": HyVideoSRModelLoader,
    "HyVideo15VaeDecode": HyVideoVaeDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HyVideo15TransformerLoader": "HunyuanVideo 1.5 Leo Transformer Model Loader",
    "HyVideo15VaeLoader": "HunyuanVideo 1.5 Leo VAE Model Loader",
    "HyVideo15TextEncoderLoader": "HunyuanVideo 1.5 Leo Text Encoder Model Loader",
    "HyVideo15VisionEncoderLoader": "HunyuanVideo 1.5 Leo Vision Encoder Model Loader",
    "HyVideo15Byt5Loader": "HunyuanVideo byt5 loader",
    "HyVideo15TextEncode": "HunyuanVideo text encode",
    "HyVideo15CFG": "HunyuanVideo CFG",
    "HyVideo15GlyphByT5": "HunyuanVideo Glyph by T5",
    "HyVideo15VisionEncode": "HunyuanVideo Vision Encode",
    "HyVideo15VaeEncode": "HunyuanVideo VAE Encode",
    "HyVideo15Transformer": "HunyuanVideo Transformer",
    "HyVideo15LatentsPrepare": "HunyuanVideo Prepare Latents",
    "HyVideo15SR": "HunyuanVideo Super Resolution",
    "HyVideo15SRModelLoader": "HunyuanVideo Super Resolution Model Loader",
    "HyVideo15VaeDecode": "HunyuanVideo Vae Decode",
}
