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
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)



import comfy.utils
import torch
import json
import loguru
from PIL import Image
import numpy as np

import subprocess
from hyvideo.commons import PRECISION_TO_TYPE
from hyvideo.models.autoencoders import hunyuanvideo_15_vae
from hyvideo.models.vision_encoder import VisionEncoder
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.models.text_encoders import TextEncoder, PROMPT_TEMPLATE

from hyvideo.utils.data_utils import generate_crop_size_list, get_closest_ratio
from hyvideo.commons import is_sparse_attn_supported, is_flash3_available


import folder_paths
from hyvideo.models.autoencoders import hunyuanvideo_15_vae
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
from hyvideo.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat

from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer,
)

from hyvideo import __initialize_default_distributed_environment

from hyvideo.commons import TRANSFORMER_VERSION_TO_SR_VERSION

__initialize_default_distributed_environment()


models_root = os.path.dirname(folder_paths.get_folder_paths("checkpoints")[0])
folder_paths.add_model_folder_path("hyvideo-1.5", models_root, "hyvideo-1.5")




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



class HyVideo15T2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hunyuanvideo_model_config": ("HUNYUANVIDEO_MODEL_CONFIG",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A close-up shot captures a scene on a polished, light-colored granite kitchen counter, illuminated by soft natural light from an unseen window. Initially, the frame focuses on a tall, clear glass filled with golden, translucent apple juice standing next to a single, shiny red apple with a green leaf still attached to its stem. The camera moves horizontally to the right. As the shot progresses, a white ceramic plate smoothly enters the frame, revealing a fresh arrangement of about seven or eight more apples, a mix of vibrant reds and greens, piled neatly upon it. A shallow depth of field keeps the focus sharply on the fruit and glass, while the kitchen backsplash in the background remains softly blurred. The scene is in a realistic style."
                }),
                "negative_prompt": ("STRING",{
                    "multiline": True,
                    "default": ""
                }),
                "video_length": ("INT",{"default": 121}),
                "num_inference_steps": ("INT",{"default": 50}),
                "guidance_scale": ("FLOAT",{"default": 6.0}),
                "num_videos_per_prompt": ("INT",{"default": 1}),
                "output_type": ("STRING",{"default": "pt"}),
                "create_sr_pipeline": ("BOOLEAN",{"default": True}),
                "aspect_ratio": ("STRING",{"default": "16:9"}),
                "sr_num_inference_steps": ("INT",{"default": 8}),
                "prompt_rewrite":("BOOLEAN", {"default": False, "tooltip": "Rewrite prompt."}),
                "prompt_rewrite_base_url": ("STRING", {"default": ""}),
                "prompt_rewrite_model_name": ("STRING", {"default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate_text_to_video"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def generate_text_to_video(
        self,
        hunyuanvideo_model_config,
        prompt,
        video_length,
        num_inference_steps,
        guidance_scale,
        negative_prompt,
        num_videos_per_prompt,
        output_type,
        seed = 123,
        create_sr_pipeline = True,
        aspect_ratio = "16:9",
        sr_num_inference_steps = 8,
        prompt_rewrite=False,
        prompt_rewrite_base_url="",
        prompt_rewrite_model_name=""
    ):
        
        
        byt5_kwargs = hunyuanvideo_model_config["byt5_kwargs"]

        
        pipeline = HunyuanVideo_1_5_Pipeline(
            vae=hunyuanvideo_model_config["vae"],
            text_encoder=hunyuanvideo_model_config["text_encoder"],
            transformer=hunyuanvideo_model_config["transformer"],
            scheduler=hunyuanvideo_model_config["scheduler"],
            text_encoder_2=hunyuanvideo_model_config["text_encoder_2"],
            progress_bar_config=None,
            byt5_model=byt5_kwargs["byt5_model"],
            byt5_tokenizer=byt5_kwargs["byt5_tokenizer"],
            byt5_max_length=byt5_kwargs["byt5_max_length"],
            prompt_format=hunyuanvideo_model_config["prompt_format"],
            execution_device='cuda',
            vision_encoder=hunyuanvideo_model_config["vision_encoder"],
            enable_offloading=hunyuanvideo_model_config["enable_offloading"],
        )
        
        
        if hunyuanvideo_model_config["enable_offloading"]:
            device = torch.device('cpu')
        else:
            device = torch.device(hunyuanvideo_model_config["device_opt"])
        
        
        
        
        
        if create_sr_pipeline:
            sr_version = TRANSFORMER_VERSION_TO_SR_VERSION[hunyuanvideo_model_config["transformer_version"]]
            sr_pipeline = pipeline.create_sr_pipeline(os.path.join(hunyuanvideo_model_config["model_path"], "upscale_models", "hyvideo15"), sr_version, transformer_dtype=dtype_options[hunyuanvideo_model_config["transformer_dtype"]], device=device)
            pipeline.sr_pipeline = sr_pipeline
            
        os.environ['T2V_REWRITE_BASE_URL'] = prompt_rewrite_base_url
        os.environ['T2V_REWRITE_MODEL_NAME'] = prompt_rewrite_model_name
        
        result = pipeline(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            prompt_rewrite=prompt_rewrite,
            enable_sr=create_sr_pipeline,
            sr_num_inference_steps=sr_num_inference_steps,
            num_inference_steps=num_inference_steps,
            video_length=video_length,
            guidance_scale=guidance_scale, # TODO: OPTIMAL HYPER PARAM FOR SCALE AND SHIFT
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt, # TODO: kevinkhwu: test this functionality
            seed=seed,
            output_type=output_type,
            return_dict=True,
        )
        
        if output_type == "latent":
            if create_sr_pipeline:
                return result.sr_videos
            else:
                return result.videos
        
        # 格式转换： (B,C,F,H,W) -> (B,F,H,W,C)
        if create_sr_pipeline:
            video_tensor = result.sr_videos.permute(0, 2, 3, 4, 1)
        else:
            video_tensor = result.videos.permute(0, 2, 3, 4, 1)
        
        if num_videos_per_prompt == 1:
            return (video_tensor[0],)
        else:
            return (video_tensor,)

def tensor_to_pil(comfyui_tensor):
    image_np = comfyui_tensor[0].cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)

class HyVideo15I2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hunyuanvideo_model_config": ("HUNYUANVIDEO_MODEL_CONFIG",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A close-up shot captures a scene on a polished, light-colored granite kitchen counter, illuminated by soft natural light from an unseen window. Initially, the frame focuses on a tall, clear glass filled with golden, translucent apple juice standing next to a single, shiny red apple with a green leaf still attached to its stem. The camera moves horizontally to the right. As the shot progresses, a white ceramic plate smoothly enters the frame, revealing a fresh arrangement of about seven or eight more apples, a mix of vibrant reds and greens, piled neatly upon it. A shallow depth of field keeps the focus sharply on the fruit and glass, while the kitchen backsplash in the background remains softly blurred. The scene is in a realistic style."
                }),
                "negative_prompt": ("STRING",{
                    "multiline": True,
                    "default": ""
                }),

                "video_length": ("INT",{"default": 121}),
                "reference_image": ("IMAGE",),
                "num_inference_steps": ("INT",{"default": 50}),
                "guidance_scale": ("FLOAT",{"default": 6.0}),
                "num_videos_per_prompt": ("INT",{"default": 1}),
                "output_type": ("STRING",{"default": "pt"}),
                "create_sr_pipeline": ("BOOLEAN",{"default": True}),
                "aspect_ratio": ("STRING",{"default": "16:9"}),
                "sr_num_inference_steps": ("INT",{"default": 8}),
            },
            "optional": {
                "seed": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate_image_to_video"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def generate_image_to_video(
        self,
        hunyuanvideo_model_config,
        prompt,
        video_length,
        reference_image,
        num_inference_steps,
        guidance_scale,
        negative_prompt,
        num_videos_per_prompt,
        output_type,
        seed = 123,
        create_sr_pipeline = True,
        aspect_ratio = "16:9",
        sr_num_inference_steps = 8,
    ):
        
        byt5_kwargs = hunyuanvideo_model_config["byt5_kwargs"]

        
        pipeline = HunyuanVideo_1_5_Pipeline(
            vae=hunyuanvideo_model_config["vae"],
            text_encoder=hunyuanvideo_model_config["text_encoder"],
            transformer=hunyuanvideo_model_config["transformer"],
            scheduler=hunyuanvideo_model_config["scheduler"],
            text_encoder_2=hunyuanvideo_model_config["text_encoder_2"],
            progress_bar_config=None,
            byt5_model=byt5_kwargs["byt5_model"],
            byt5_tokenizer=byt5_kwargs["byt5_tokenizer"],
            byt5_max_length=byt5_kwargs["byt5_max_length"],
            prompt_format=hunyuanvideo_model_config["prompt_format"],
            execution_device='cuda',
            vision_encoder=hunyuanvideo_model_config["vision_encoder"],
            enable_offloading=hunyuanvideo_model_config["enable_offloading"],
        )
        
        
        if hunyuanvideo_model_config["enable_offloading"]:
            device = torch.device('cpu')
        else:
            device = torch.device(hunyuanvideo_model_config["device_opt"])
        
        
        
        if create_sr_pipeline:
            sr_version = TRANSFORMER_VERSION_TO_SR_VERSION[hunyuanvideo_model_config["transformer_version"]]
            sr_pipeline = pipeline.create_sr_pipeline(os.path.join(hunyuanvideo_model_config["model_path"], "upscale_models", "hyvideo15"), sr_version, transformer_dtype=dtype_options[hunyuanvideo_model_config["transformer_dtype"]], device=device)
            pipeline.sr_pipeline = sr_pipeline
        
        pil_image  = tensor_to_pil(reference_image)
        
        result = pipeline(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            enable_sr=create_sr_pipeline,
            sr_num_inference_steps=sr_num_inference_steps,
            num_inference_steps=num_inference_steps,
            video_length=video_length,
            guidance_scale=guidance_scale, # TODO: OPTIMAL HYPER PARAM FOR SCALE AND SHIFT
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt, # TODO: kevinkhwu: test this functionality
            seed=seed,
            reference_image=pil_image,
            output_type=output_type,
            return_dict=True,
        )
        
        if output_type == "latent":
            if create_sr_pipeline:
                return result.sr_videos
            else:
                return result.videos
        
        if create_sr_pipeline:
            video_tensor = result.sr_videos.permute(0, 2, 3, 4, 1)
        else:
            video_tensor = result.videos.permute(0, 2, 3, 4, 1)
        
        if num_videos_per_prompt == 1:
            return (video_tensor[0],)
        else:
            return (video_tensor,)

class HyVideo15ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "model_path": ("STRING",{"default": "","tooltip": "1. If you have already manually downloaded the model to your directory, specify the model folder path hyvideo-1.5 here. 2. If not specified, it will be automatically downloaded to the default folder for continued use."}),
                "attn_mode": (["flash","ptm_sparse_attn","flash3",], {"default": "flash"}),
                "byt5_max_length": ("INT",{"default": 256}),
                "vision_encoder_type": ("STRING",{"default": "siglip"}),
                "vision_encoder_precision": ("STRING",{"default": "fp16"}),
                "text_encoder_type": ("STRING",{"default": "qwen-2.5vl-7b"}),
                "text_encoder_tokenizer_type": ("STRING",{"default": "qwen-2.5vl-7b"}),
                "text_encoder_max_length": ("INT",{"default": 1000}),
                "text_encoder_precision": ("STRING",{"default": "fp16"}),
                "text_encoder_hidden_state_skip_layer": ("INT",{"default": 2}),
                "text_encoder_apply_final_norm": ("BOOLEAN",{"default": False}),
                "text_encoder_reproduce": ("BOOLEAN",{"default": False}),
                "resolution": (["480p","720p",], {"default": "480p"}),
                "task": (["t2v","i2v",], {"default": "t2v"}),
                "enable_offloading": ("BOOLEAN",{"default": False}),
                "enable_group_offloading": ("BOOLEAN",{"default": False}),
                "transformer_dtype": (["float32","float64","float16","bfloat16","uint8","int8","int16","int32","int64"],
                                      {"default": "bfloat16"}),
                "device_opt": (["cpu","cuda",], {"default": "cuda"}),
                "hf_token": ("STRING",{"default": ""}),
                "flow_shift": ("FLOAT",{"default": None, "tooltip": "When the resolution is 480p, the recommended shift value is 5, and when the resolution is 720p, the recommended shift value is 7. If you do not set this value, it will be automatically configured according to the recommendations of this rule."}),
            }
        }

    RETURN_TYPES = ("HUNYUANVIDEO_MODEL_CONFIG",)
    RETURN_NAMES = ("hunyuanvideo_model_config",)
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def loadmodel(self,model_path,attn_mode="flash",byt5_max_length=256,vision_encoder_type="siglip",vision_encoder_precision="fp16",
                  text_encoder_type="qwen-2.5vl-7b",text_encoder_tokenizer_type="qwen-2.5vl-7b",text_encoder_max_length=1000,
                  text_encoder_precision="fp16",text_encoder_hidden_state_skip_layer=2,text_encoder_apply_final_norm=False,
                  text_encoder_reproduce=False,resolution="480p",task="t2v",enable_offloading=False,enable_group_offloading=False,
                  transformer_dtype="bfloat16",device_opt="cuda",hf_token="",flow_shift=None):
        
    
        if flow_shift is None or flow_shift == 0:
            if resolution == "480p":
                flow_shift = 5.0
            else:
                flow_shift = 7.0
        if model_path == "":
            self._download(hf_token)
            model_path = folder_paths.models_dir
        
        if enable_group_offloading is None:
            assert enable_offloading is None
            offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
            enable_offloading = offloading_config['enable_offloading']
            enable_group_offloading = offloading_config['enable_group_offloading']


        if enable_offloading:
            device = torch.device('cpu')
        else:
            device = torch.device(device_opt)

        byt5_kwargs, prompt_format = HunyuanVideo_1_5_Pipeline._load_byt5(os.path.join(model_path, "text_encoders", "hyvideo15"), True, byt5_max_length, device=device)
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=True,
            solver="euler",
        )
        
        vision_encoder = VisionEncoder(
            vision_encoder_type=vision_encoder_type,
            vision_encoder_precision=vision_encoder_precision,
            vision_encoder_path=os.path.join(model_path, "clip_vision", "hyvideo15", "siglip"),
            processor_type=None,
            processor_path=None,
            output_key=None,
            logger=loguru.logger,
            device='cuda'
        )
        
        text_encoder = TextEncoder(
            text_encoder_type=text_encoder_type,
            tokenizer_type=text_encoder_tokenizer_type,
            text_encoder_path=os.path.join(model_path, "text_encoder", "hyvideo15", "llm"),
            max_length=text_encoder_max_length,
            text_encoder_precision=text_encoder_precision,
            prompt_template=PROMPT_TEMPLATE['li-dit-encode-image-json'],
            prompt_template_video=PROMPT_TEMPLATE['li-dit-encode-video-json'],
            hidden_state_skip_layer=text_encoder_hidden_state_skip_layer,
            apply_final_norm=text_encoder_apply_final_norm,
            reproduce=text_encoder_reproduce,
            logger=loguru.logger,
            device='cuda',

        )
        text_encoder_2 = None
        
        vae = hunyuanvideo_15_vae.AutoencoderKLConv3D.from_pretrained(os.path.join(model_path, "vae", "hyvideo15")).to(device)
        
        
        transformer_version = f'{resolution}_{task}'
        
        
        transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(os.path.join(model_path, "diffusion_models", "hyvideo15", transformer_version), torch_dtype=dtype_options[transformer_dtype]).to(device)
        transformer.set_attn_mode(attn_mode)
        
        if enable_group_offloading:
            assert enable_offloading
            transformer.enable_group_offload(onload_device=torch.device('cuda'), num_blocks_per_group=4)
        
        

        out_put = {
            "vae": vae,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "transformer_version": transformer_version,
            "scheduler": scheduler,
            "byt5_kwargs": byt5_kwargs,
            "prompt_format": prompt_format,
            "text_encoder_2": text_encoder_2,
            "vision_encoder": vision_encoder,
            "transformer_dtype": transformer_dtype,
            "enable_offloading": enable_offloading,
            "device_opt": device_opt,
            "model_path": model_path,
            
        }
        return (out_put,)

    def _download(self, hf_token):
        path = os.path.join(folder_paths.models_dir, "upscale_models", "hyvideo15")
        if not os.path.exists(path):
            tmp_path = folder_paths.get_temp_directory()
            self._cmd(f"hf download tencent/HunyuanVideo-1.5 --include \"upsampler/*\" --local-dir {tmp_path}")
            self._cmd(f"mv {tmp_path}/upsampler {path}")

        path = os.path.join(folder_paths.models_dir, "diffusion_models", "hyvideo15")
        if not os.path.exists(path):
            tmp_path = folder_paths.get_temp_directory()
            self._cmd(f"hf download tencent/HunyuanVideo-1.5 --include \"transformer/*\" --local-dir {tmp_path}")
            self._cmd(f"mv {tmp_path}/transformer {path}")
        
        path = os.path.join(folder_paths.models_dir, "vae", "hyvideo15")
        if not os.path.exists(path):
            tmp_path = folder_paths.get_temp_directory()
            self._cmd(f"hf download tencent/HunyuanVideo-1.5 --include \"vae/*\" --local-dir {tmp_path}")
            self._cmd(f"mv {tmp_path}/vae {path}")

        path = os.path.join(folder_paths.models_dir, "text_encoders", "hyvideo15", "llm")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            self._cmd(f"hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir {path}")
        
        path = os.path.join(folder_paths.models_dir, "clip_vision", "hyvideo15", "siglip")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            self._cmd(f"hf download black-forest-labs/FLUX.1-Redux-dev --local-dir {path} --token {hf_token}")

        path = os.path.join(folder_paths.models_dir, "text_encoders", "hyvideo15")
        byt5_path = os.path.join(path, "byt5-small")
        glyph_path = os.path.join(path, "Glyph-SDXL-v2")
        if not os.path.exists(byt5_path):
            self._cmd(f"hf download google/byt5-small --local-dir {path}")
        if not os.path.exists(glyph_path):
            self._cmd(f"modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir {path}")

        
    def _cmd(self,cmd):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True,check=True)
    
    def _ensure_directories(path_list):
        for path in path_list:
            if not os.path.exists(path):
                os.makedirs(path)

    def get_model_dir_path(model_dir):
        all_paths = folder_paths.get_folder_paths(model_dir)
        for path in all_paths:
            if "ComfyUI/models" in path.replace("\\", "/"):
                return path
        return all_paths[0] if all_paths else ""

NODE_CLASS_MAPPINGS = {
    "HyVideo15T2VSampler": HyVideo15T2VSampler,
    "HyVideo15I2VSampler": HyVideo15I2VSampler,
    "HyVideo15ModelLoader": HyVideo15ModelLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HyVideo15T2VSampler": "HunyuanVideo 1.5 Leo Text-to-Video Sampler",
    "HyVideo15I2VSampler": "HunyuanVideo 1.5 Leo Image-to-Video Sampler",
    "HyVideo15ModelLoader": "HunyuanVideo 1.5 Model Loader",
}
