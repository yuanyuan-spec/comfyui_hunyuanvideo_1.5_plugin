import os
import argparse
from modelscope import snapshot_download
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download


def hf_download_model(repo_id, allow_patterns, local_dir,hf_token=None):

    hf_snapshot_download(
        repo_id=repo_id, # "tencent/HunyuanVideo-1.5"
        allow_patterns=allow_patterns, # f"transformer/{transformer_version}/*"
        local_dir=local_dir,
        token=hf_token
    )

def ms_download_model(model_id, local_dir):

    ms_snapshot_download(
        model_id=model_id,
        cache_dir=local_dir,  # 相当于 --local_dir
    )
    

    
    
    
