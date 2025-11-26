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

from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer,
)
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
import os
import torch.nn.functional as F

from node_adv import tensor_to_pil
import folder_paths

from node_adv import HunyuanVideoPipelineOutput
import shutil


from hyvideo.commons import (
    PRECISION_TO_TYPE, auto_offload_model
)
from hyvideo.models.autoencoders import hunyuanvideo_15_vae
from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
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

from node_adv import dtype_options,get_immediate_subdirectories,get_model_dir_path,run_cmd

import comfy.model_management as mm
from hyvideo.pipelines.hunyuan_video_sr_pipeline import expand_dims,BucketMap,SizeMap

class HyVideoSrLatentsPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hyvid_cfg": ("HYVID15CFG", ),
                "vae": ("HYVID15VAE", ),
                "aspect_ratio": ("STRING", ),
                "target_dtype": (["float32","float64","float16","bfloat16","uint8","int8","int16","int32","int64"], {"default": "bfloat16"}),
            },
            "optional": {
                "transformer_latent": ("HYVID15TRANSFORMERLATENT", ),
                "reference_image": ("IMAGE", {"default": None} ),
            }
        }
    RETURN_TYPES = ("HYVID15LATENTSDICT", "INT", "INT", "INT")
    RETURN_NAMES = ("latents_dict", "height", "width", "n_tokens")
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def process(self, hyvid_cfg, vae, aspect_ratio,target_dtype,transformer_latent=None,reference_image=None):
        device = mm.get_torch_device()
        
        self.vae = vae
        self.target_size_config = {
            "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
            "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
            "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
            "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
        }
        self.scheduler = FlowMatchDiscreteScheduler(
            shift=hyvid_cfg.get("flow_shift", 5.0),
            reverse=True,
            solver="euler",
        )
        
        ideal_resolution = hyvid_cfg["sr_transformer_config"].ideal_resolution
        
        base_resolution = hyvid_cfg["transformer_config"].ideal_resolution
        
        lq_latents = transformer_latent

        sr_stride = 16
        base_size = SizeMap[base_resolution]
        sr_size = SizeMap[ideal_resolution]
        bucket_map = BucketMap(lr_base_size=base_size, hr_base_size=sr_size, lr_patch_size=16, hr_patch_size=sr_stride)
        lr_video_height, lr_video_width = [x * 16 for x in lq_latents.shape[-2:]]
        width, height = bucket_map((lr_video_width, lr_video_height))

        latent_target_length, latent_height, latent_width = self._get_latent_size(hyvid_cfg["video_length"], height, width, False)
        n_tokens = latent_target_length * latent_height * latent_width

        num_channels_latents = hyvid_cfg["transformer_config"].in_channels
        latents = self._prepare_latents(
            hyvid_cfg["batch_size"] * hyvid_cfg["num_videos_per_prompt"],
            32,
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

    def _get_latent_size(self, video_length, height, width, reorg_token):
        spatial_compression_ratio = self._vae_spatial_compression_ratio
        temporal_compression_ratio = self._vae_temporal_compression_ratio
        video_length = (video_length - 1) // temporal_compression_ratio + 1
        if reorg_token:
            video_length = (video_length + 1) // 2
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

class HyVideoSrVaeEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("HYVID15VAE", ),
                "latents_dict": ("HYVID15LATENTSDICT", ),
                "height": ("INT", {"default": 768}),
                "width": ("INT", {"default": 512}),
                "hyvid_cfg": ("HYVID15CFG", ),
                "upsampler": ("UPASAMPLER", )
            },
            "optional": {
                "reference_image": ("IMAGE", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("HYVID15VAECONCAT",)
    RETURN_NAMES = ("vae_concat")
    FUNCTION = "encode"
    CATEGORY = "HunyuanVideoWrapper1.5"
    def encode(self, vae, latents_dict,height, width, hyvid_cfg, upsampler,reference_image=None):
        device = torch.device('cuda')
        enable_offloading = False
        self.device = device
        self.vae = vae
        
        if reference_image is not None:
            reference_image = tensor_to_pil(reference_image)

        with auto_offload_model(vae, device, enabled=enable_offloading): 
            cond_latents = self._get_image_condition_latents(reference_image, height, width)  

        task_type = "i2v" if reference_image is not None else "t2v"
        multitask_mask = self._get_task_mask(task_type, latents_dict["latent_target_length"])
        
        

        lq_latents = latents_dict["latents"]
        
        tgt_shape = latents_dict["latents"].shape[-2:]  # (h w)
        bsz = lq_latents.shape[0]
        lq_latents = rearrange(lq_latents, "b c f h w -> (b f) c h w")
        lq_latents = F.interpolate(lq_latents, size=tgt_shape, mode="bilinear", align_corners=False)
        lq_latents = rearrange(lq_latents, "(b f) c h w -> b c f h w", b=bsz)
        with auto_offload_model(upsampler, device, enabled=enable_offloading):
            lq_latents = upsampler(lq_latents.to(dtype=torch.float32, device=device))
        lq_latents = lq_latents.to(dtype=latents_dict["latents"].dtype)
        # lq_latents = lq_latents * self.vae.config.scaling_factor

        noise_scale = 0.7
        lq_latents = self.add_noise_to_lq(lq_latents, noise_scale)
        condition = self._get_condition(latents_dict["latents"], lq_latents, cond_latents, task_type) # TODO: image cond?
        c = lq_latents.shape[1]

        zero_condition = condition.clone()
        zero_condition[:, c + 1 : 2 * c + 1] = torch.zeros_like(lq_latents)
        zero_condition[:, 2 * c + 1] = 0

        vae_concat = {
            "zero_condition": zero_condition,
            "condition": condition,
            "noise_scale": noise_scale,
            "task_type": task_type
        }
        return (vae_concat,)
    
    def _get_condition(self, latents, lq_latents, img_cond, task):
        """
        latents: shape (b c f h w)
        """
        b, c, f, h, w = latents.shape
        cond = torch.zeros([b, c * 2 + 2, f, h, w], device=latents.device, dtype=latents.dtype)

        cond[:, c + 1 : 2 * c + 1] = lq_latents
        cond[:, 2 * c + 1] = 1
        if "t2v" in task:
            return cond
        elif "i2v" in task:
            cond[:, :c, :1] = img_cond
            cond[:, c + 1, 0] = 1
            return cond
        else:
            raise ValueError(f"Unsupported task: {task}")
    

    def _get_image_condition_latents(self, reference_image, height, width):
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
            
            ref_images_pixel_values = ref_image_transform(reference_image).unsqueeze(0).unsqueeze(2).to(self.device)
            
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
    
    def add_noise_to_lq(self, lq_latents, strength=0.7):
        noise = torch.randn_like(lq_latents)
        timestep = torch.tensor([1000.0], device=self.device) * strength
        t = expand_dims(timestep, lq_latents.ndim)
        return (1 - t / 1000.0) * lq_latents + (t / 1000.0) * noise


class HyVideoSrTransformer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hyvid_cfg": ("HYVID15CFG", ),
                "target_dtype": (["float32","float64","float16","bfloat16","uint8","int8","int16","int32","int64"], {"default": "bfloat16"}),
                "n_tokens": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 1000, "step": 1}),
                "transformer": ("HYVID15TRANSFORMER", ),
                "vae_concat": ("HYVID15VAECONCAT", ),
                "hyvid_embeds": ("HYVIDEMBEDS", ),
                "vision_states": ("HYVID15VISIONSTATES", ),
                "extra_kwargs": ("HYVID15EXTRAKWARGS", ),
                "lantens_dict": ("HYVID15LATENTSDICT", ),
            },
            "optional": {
                "embedded_guidance_scale": ("FLOAT", {"default": None, "min": 0, "max": 10, "step": 0.01}), #self.config.embedded_guidance_scale
                "autocast_enabled": ("BOOLEAN", {"default": True}),
                "eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01} ),
                "guidance_rescale" : ("FLOAT", {"default": 0.0}),
            }
        }
    RETURN_TYPES = ("HYVID15TRANSFORMERLATENT", )
    RETURN_NAMES = ("latent",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper1.5"
    def process(self, hyvid_cfg,target_dtype, n_tokens, steps, transformer, vae_concat, hyvid_embeds, vision_states, extra_kwargs,lantens_dict,
                embedded_guidance_scale=None, autocast_enabled=True,eta=0.0,guidance_rescale=0.0):
        
        device = torch.device('cuda')
        enable_offloading = False
        
        extra_step_kwargs = self._prepare_extra_func_kwargs(
            hyvid_cfg["scheduler"].step, {"generator": hyvid_cfg["generator"], "eta": eta},
        )

        extra_set_timesteps_kwargs = self._prepare_extra_func_kwargs(
            hyvid_cfg["scheduler"].set_timesteps, {"n_tokens": n_tokens}
        )
        
        latents = lantens_dict["latents"]
        
        
        timesteps, num_inference_steps = retrieve_timesteps(
            hyvid_cfg["scheduler"],
            steps,
            device,
            **extra_set_timesteps_kwargs,
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * hyvid_cfg["scheduler"].order
        self._num_timesteps = len(timesteps)
        
        condition = vae_concat["condition"]

        with auto_offload_model(transformer, device, enabled=enable_offloading):
            progress_bar = self.progress_bar(total=num_inference_steps) 
            self.do_classifier_free_guidance = hyvid_cfg["guidance_scale"] > 1
            for i, t in enumerate(timesteps):
                if t < 1000 *vae_concat["noise_scale"]:
                    condition = vae_concat["zero_condition"]

                latents_concat = torch.concat([latents, condition], dim=1)
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
                
                def to_device(obj):
                    return obj.to(device) if torch.is_tensor(obj) else obj

                latent_model_input = to_device(latent_model_input)
                t_expand = to_device(t_expand)
                hyvid_embeds["prompt_embeds"] = to_device(hyvid_embeds["prompt_embeds"])
                hyvid_embeds["prompt_embeds_2"] = to_device(hyvid_embeds["prompt_embeds_2"])
                hyvid_embeds["prompt_mask"] = to_device(hyvid_embeds["prompt_mask"])
                timesteps_r = to_device(timesteps_r)
                vision_states = to_device(vision_states)
                guidance_expand = to_device(guidance_expand)

                with torch.autocast(device_type=str(device), dtype=dtype_options[target_dtype], enabled=autocast_enabled):
                    output = transformer(
                        latent_model_input,
                        t_expand,
                        hyvid_embeds["prompt_embeds"],
                        hyvid_embeds["prompt_embeds_2"],
                        hyvid_embeds["prompt_mask"],
                        timestep_r=timesteps_r,
                        vision_states=vision_states,
                        mask_type=vae_concat["task_type"],
                        guidance=guidance_expand,
                        return_dict=False,
                        extra_kwargs=extra_kwargs,
                    )
                    noise_pred = output[0]


                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_rescale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = hyvid_cfg["scheduler"].step(noise_pred, t, latents, extra_step_kwargs, return_dict=False)[0]

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % hyvid_cfg["scheduler"].order == 0):
                    if progress_bar is not None:
                        progress_bar.update(1)

        return (latents,)
    
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
    
    def progress_bar(self, total, desc="Processing"):
        """使用ComfyUI进度条"""
        pbar = comfy.utils.ProgressBar(total)
        return pbar
    
    
class HyVidelSrTransformerUpsamplerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upsampler_path": (get_immediate_subdirectories(get_model_dir_path("upscale_models")),),
                "transformer_path": (get_immediate_subdirectories(os.path.join(folder_paths.models_dir,"diffusion_models")),),
                "resolution": (["480p", "720p", "1080p"], {"default": "480p"}),
                "task_type": (["t2v", "i2v"], {"default": "i2v"}),
            },
            "optional": {
                "transformer_dtype": (["float32","float64","float16","bfloat16","uint8","int8","int16","int32","int64"], {"default":    "bfloat16"}),
            }
        }
        
    RETURN_TYPES = ("HYVID15TRANSFORMER", "HYVID15TRANSFORMERCONFIG","UPASAMPLER")
    RETURN_NAMES = ("model", "config", "upsampler")
    FUNCTION = "load_sr_transformer_upsampler"
    CATEGORY = "HunyuanVideoWrapper1.5"

    def load_sr_transformer_upsampler(cls, upsampler_path,transformer_path, resolution,task_type,transformer_dtype="bfloat16"):
        device = torch.device('cuda')
        
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
        transformer_version = f"{resolution}_{task_type}"
        
        sr_version = TRANSFORMER_VERSION_TO_SR_VERSION[transformer_version]
        
        if upsampler_path == "None":
            upsampler_path = os.path.join(folder_paths.models_dir, "upscale_models", "hyvideo15")
            if not os.path.exists(upsampler_path):
                tmp_path = folder_paths.get_temp_directory()
                ret = run_cmd(f"hf download tencent/HunyuanVideo-1.5 --include \"upsampler/*\" --local-dir {tmp_path}")
                loguru.logger.info(ret)
                shutil.move(os.path.join(tmp_path, "upsampler"), upsampler_path)
                # run_cmd(f"mv {tmp_path}/upsampler {upsampler_path}")

        if transformer_path == "None":
            transformer_path = os.path.join(folder_paths.models_dir, "diffusion_models", "hyvideo15")
            if not os.path.exists(os.path.join(transformer_path,sr_version)):
                os.makedirs(transformer_path, exist_ok=True)
                tmp_path = folder_paths.get_temp_directory()
                run_cmd(f"hf download tencent/HunyuanVideo-1.5 --include \"transformer/{sr_version}/*\" --local-dir {tmp_path}")
                shutil.move(os.path.join(tmp_path, "transformer",sr_version), transformer_path)
                # run_cmd(f"mv {tmp_path}/transformer {transformer_path}")

        transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(os.path.join(transformer_path,sr_version), torch_dtype=dtype_options[transformer_dtype]).to(device)
        upsampler_cls = SRTo720pUpsampler if "720p" in sr_version else SRTo1080pUpsampler
        upsampler = upsampler_cls.from_pretrained(os.path.join(upsampler_path,  sr_version)).to(device)
        return (transformer, transformer.config,upsampler,)

class HyVidelSrVaeDecoder:
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
            }
        }
    RETURN_TYPES = ("HYVID15SROUT", )
    RETURN_NAMES = ("sr_out",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper1.5"
    def process(self, latents, output_type, vae, hyvid_cfg, enable_offloading=True, vae_dtype="bfloat16",):
        device = torch.device('cuda')
        enable_offloading =False
        

        self.vae = vae
        self.enable_offloading = enable_offloading
        self.vae_dtype = dtype_options[vae_dtype]
        self.vae_autocast_enabled = True
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

            # if hyvid_cfg["reorg_token"]:
            #     latents = rearrange(latents, 'b c f h w -> b f c h w')
            #     latents = rearrange(latents, 'b f (n c) h w -> b (f n) c h w', n=2)
            #     latents = rearrange(latents, 'b f c h w -> b c f h w')
            #     latents = latents[:, :, 1:]

            if hasattr(self.vae, 'enable_tile_parallelism'):
                self.vae.enable_tile_parallelism()

            with torch.autocast(device_type="cuda", dtype=self.vae_dtype, enabled=self.vae_autocast_enabled), auto_offload_model(self.vae, device, enabled=self.enable_offloading):
                self.vae.enable_tiling()
                video_frames = self.vae.decode(latents, return_dict=False, generator=hyvid_cfg["generator"])[0]
                self.vae.disable_tiling()

            if video_frames is not None:
                video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()


        result = HunyuanVideoPipelineOutput(videos=video_frames)
        
        return result

NODE_CLASS_MAPPINGS = {
    "HyVideoSrLatentsPrepare": HyVideoSrLatentsPrepare,
    "HyVideoSrVaeEncode": HyVideoSrVaeEncode,
    "HyVideoSrTransformer": HyVideoSrTransformer,
    "HyVidelSrTransformerUpsamplerLoader": HyVidelSrTransformerUpsamplerLoader,
    "HyVidelSrVaeDecoder": HyVidelSrVaeDecoder,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HyVideoSrLatentsPrepare": "HunyuanVideo Sr Latents PrePare",
    "HyVideoSrVaeEncode": "HunyuanVideo Sr Vae Encode",
    "HyVideoSrTransformer": "HunyuanVideo Sr Transformer",
    "HyVidelSrTransformerUpsamplerLoader": "HyVidelSrTransformerUpsamplerLoader",
    "HyVidelSrVaeDecoder": "HyVidelSrVaeDecoder",
}