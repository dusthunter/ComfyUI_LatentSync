# ComfyUI-LatentSync

ComfyUI-LatentSync 是一个用于 ComfyUI 的唇型同步(lip sync)节点模块，基于字节开源的LatentSync v1.5 实现高质量的音视频唇形同步。

## 功能特性

- 基于LatentSync实现潜空间的高质量唇型同步
- 支持图片和视频的唇型同步处理
- 采取视频质量优先的视频合成策略

## 模型Checkpoints目录结构

```
checkpoints/
├── auxiliary
│   ├── 2DFAN4-cd938726ad.zip
│   ├── i3d_torchscript.pt
│   ├── koniq_pretrained.pkl
│   ├── models
│   │   ├── buffalo_l
│   │   │   ├── 1k3d68.onnx
│   │   │   ├── 2d106det.onnx
│   │   │   ├── det_10g.onnx
│   │   │   ├── genderage.onnx
│   │   │   └── w600k_r50.onnx
│   │   └── buffalo_l.zip
│   ├── s3fd-619a316812.pth
│   ├── sfd_face.pth
│   ├── syncnet_v2.model
│   ├── vgg16-397923af.pth
│   └── vit_g_hybrid_pt_1200e_ssv2_ft.pth
├── config.json
├── latentsync_unet.pt
├── stabilityai
│   └── sd-vae-ft-mse
│       ├── config.json
│       ├── diffusion_pytorch_model.bin
│       └── diffusion_pytorch_model.safetensors
├── stable_syncnet.pt
└── whisper
    └── tiny.pt
```

## 安装说明

1. 首先确保已安装 ComfyUI

2. 克隆本仓库到 ComfyUI 的 custom_nodes 目录:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dusthunter/ComfyUI-LatentSync.git
```

3. 安装依赖:
```bash
cd ComfyUI-LatentSync
pip install -r requirements.txt
```

4. 下载预训练模型和权重文件到 checkpoints 目录
具体路径参考：[LatentSync 1.5](https://github.com/bytedance/LatentSync)

## 使用方法

1. 启动 ComfyUI 后，你会在节点列表中找到 LatentSync 相关节点

2. 基本工作流程:
   - 加载源图片/视频和音频文件
   - 使用 LatentSync 节点进行处理
   - 导出生成的结果

3. 预设工作流:
   - workflow/image_latentsync_v1.5.json: 图片唇型同步工作流
   - workflow/video_latentsync_v1.5.json: 视频唇型同步工作流

## 致谢

这是一个LatentSync非官方的ComfyUI实现：
- [LatentSync 1.5](https://github.com/bytedance/LatentSync)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## 许可证

本项目采用沿用LatentSync授权模式，采用 Apache 2.0 许可证授权 - 详情请参阅相关许可证文件。
