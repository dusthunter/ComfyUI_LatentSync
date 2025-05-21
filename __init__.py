import os
import sys

# 获取当前文件 (__init__.py) 所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# latentsync_dir = os.path.join(current_dir, "latentsync")
# sys.path.insert(0, latentsync_dir)

from .latentsync_nodes import LatentSyncNode, VideoLengthAdjuster,SaveLipSyncVideo

# 添加 WEB_DIRECTORY 变量
WEB_DIRECTORY = ("./web")

# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LatentSyncNode": LatentSyncNode,
    "VideoLengthAdjuster": VideoLengthAdjuster,
    "SaveLipSyncVideo": SaveLipSyncVideo,
}

# Display Names for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSyncNode": "LatentSync 1.5",
    "VideoLengthAdjuster": "Video Length Adjuster",
    "SaveLipSyncVideo": "Save Lip Sync Video",
 }

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS','WEB_DIRECTORY']