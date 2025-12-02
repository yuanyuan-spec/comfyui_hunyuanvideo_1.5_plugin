# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os
from dataclasses import dataclass

import torch.distributed as dist

@dataclass
class ParallelDims:
    sp: int = 1
    world_size: int = -1

    def __post_init__(self):
        if self.world_size == -1:
            if dist.is_initialized():
                self.world_size = dist.get_world_size()
            else:
                self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        # 单机单卡环境，直接设置world_mesh为None
        self.world_mesh = None
        # 创建sp_mesh的模拟对象
        self._sp_mesh = _MockMesh()

    def build_mesh(self, device_type):
        # 单机单卡环境，不需要构建mesh，直接返回None
        self.world_mesh = None
        return None

    @property
    def sp_enabled(self):
        return self.sp > 1

    @property
    def sp_group(self):
        return None

    @property
    def sp_mesh(self):
        # 返回模拟的mesh对象
        return self._sp_mesh

    @property
    def sp_rank(self):
        # 单机单卡返回0
        return 0

    @property
    def dp_enabled(self):
        return self.sp > 1


class _MockMesh:
    """模拟mesh对象的类，用于保持接口兼容"""
    def get_group(self):
        # 返回None，表示没有有效的进程组
        return None
    
    def get_local_rank(self):
        return 0

__parallel_dims = None

def initialize_parallel_state(
    sp: int = 1,
):
    global __parallel_dims
    __parallel_dims = ParallelDims(sp=sp)
    return __parallel_dims

def get_parallel_state():
    if __parallel_dims is None:
        # create default parallel states (without enabling any parallelism)
        initialize_parallel_state()
    return __parallel_dims