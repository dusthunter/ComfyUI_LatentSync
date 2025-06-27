import os
import shutil
import subprocess
import tempfile
import uuid
import imageio
import torch
import numpy as np
import torchaudio
import math
from omegaconf import OmegaConf
import folder_paths
from .latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from .latentsync.whisper.audio2feature import Audio2Feature
from .latentsync.models.unet import UNet3DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler

# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "configs/unet/stage2.yaml")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints/latentsync_unet.pt")


class LatentSyncNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "AUDIO": ("AUDIO",),
                "seed": (
                    "INT",
                    {"default": 1247, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),  # 更好的随机种子范围
                "sync_frames": (
                    "INT",
                    {"default": 16, "min": 2, "max": 30, "step": 2},
                ),  # 同步间隔帧数
                "guidance_scale": (
                    "FLOAT",
                    {"default": 2, "min": 1.0, "max": 3.0, "step": 0.1},
                ),
                "inference_steps": (
                    "INT",
                    {"default": 25, "min": 1, "max": 100, "step": 1},
                ),  # 常用范围
                "face_detect_once": (
                    "BOOLEAN",
                    {"default": True},
                ),  # 新增参数：是否只进行一次人脸检测
                "movement_threshold": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),  # 新增参数：运动阈值
            }
        }

    CATEGORY = "LatentSync"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "inference"

    # 移除未使用的 process_batch 方法

    def inference(self, IMAGE, AUDIO, seed, sync_frames, guidance_scale=1.5, inference_steps=20, face_detect_once=True, movement_threshold=0.0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_fp16_supported = (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        )
        dtype = torch.float16 if is_fp16_supported else torch.float32

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # 转换图像格式
        if torch.is_tensor(IMAGE):
            # 确保图像在正确的范围内 (0-255)
            IMAGE = (IMAGE * 255).byte()
            # 转换为 numpy 数组
            IMAGE = IMAGE.cpu().numpy()
            # 调整通道顺序从 RGB 到 BGR (如果需要)
            # images = images[..., ::-1]

        # 处理音频波形
        # ComfyUI 提供的 AUDIO 是 {"waveform": waveform, "sample_rate": sample_rate}
        # waveform 是 BATCH x Channels x Samples 或 Channels x Samples
        waveform = AUDIO["waveform"].squeeze(0)  # 移除 Batch 维度
        sample_rate = int(AUDIO["sample_rate"])

        # 将音频移动到 CPU (torchaudio.save 需要 CPU tensor)
        waveform_cpu = waveform.cpu()

        # 如果采样率不是 16000，进行重采样 (inference 脚本可能需要 16k)
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            print(f"Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz")
            try:
                # 确保 resampler 在 CPU 上操作，因为输入已经在 CPU
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=target_sample_rate
                ).cpu()
                waveform_resampled_cpu = resampler(waveform_cpu)
                sample_rate = target_sample_rate
                waveform_cpu = waveform_resampled_cpu
            except Exception as e:
                print(f"Warning: Could not resample audio: {e}")
                # 继续使用原始音频，但推理可能会失败

        # 加载配置文件
        config = OmegaConf.load(CONFIG_PATH)
        scheduler = DDIMScheduler.from_pretrained(os.path.join(BASE_DIR,"configs"))

        if config.model.cross_attention_dim == 768:
            whisper_model_path = os.path.join(BASE_DIR, "checkpoints/whisper/small.pt")
        elif config.model.cross_attention_dim == 384:
            whisper_model_path = os.path.join(BASE_DIR, "checkpoints/whisper/tiny.pt")
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")

        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda",
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )

        vae = AutoencoderKL.from_pretrained(
            os.path.join(BASE_DIR, "checkpoints/stabilityai/sd-vae-ft-mse"),
            torch_dtype=dtype,
        )
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

        denoising_unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            CHECKPOINT_PATH,
            device="cpu",
        )

        denoising_unet = denoising_unet.to(dtype=dtype)

        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        ).to("cuda")

        # 返回图像张量
        return pipeline(
            video_frames_uint8_np=IMAGE,
            audio_waveform_float_tensor=waveform_cpu,
            audio_sample_rate=target_sample_rate,
            num_frames=sync_frames,
            video_fps=config.data.video_fps,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=os.path.join(BASE_DIR,config.data.mask_image_path),
            face_detect_once=face_detect_once,
            movement_threshold=movement_threshold,  # 新增参数传递
        )


class VideoLengthAdjuster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "AUDIO": ("AUDIO",),
                "mode": (
                    ["normal", "pingpong", "loop_to_audio"],
                    {"default": "normal"},
                ),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
                "silent_padding_sec": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
            }
        }

    CATEGORY = "LatentSync"

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("IMAGE", "AUDIO")
    FUNCTION = "adjust"

    def adjust(self, IMAGE, AUDIO, mode, fps=25.0, silent_padding_sec=0.5):
        # 确保输入图像是张量
        if isinstance(IMAGE, list):
            original_frames_tensor = torch.stack(IMAGE)
        else:
            original_frames_tensor = IMAGE

        # 转换为列表方便处理
        original_frames = [
            original_frames_tensor[i] for i in range(original_frames_tensor.shape[0])
        ]

        # 处理音频波形
        waveform = AUDIO["waveform"].squeeze(0)  # 移除 Batch 维度
        sample_rate = int(AUDIO["sample_rate"])

        # 创建静音填充
        padding_samples = math.ceil(silent_padding_sec * sample_rate)
        silence = torch.zeros(
            waveform.shape[0], padding_samples, device=waveform.device
        )

        # 添加前后静音填充
        padded_waveform = torch.cat([silence, waveform, silence], dim=1)

        # 计算目标帧数
        total_audio_duration = padded_waveform.shape[1] / sample_rate  # 秒
        target_frames = math.ceil(total_audio_duration * fps)

        adjusted_frames = []

        if mode == "normal":
            # 循环模式：重复原始序列
            num_original_frames = len(original_frames)
            while len(adjusted_frames) < target_frames:
                remaining_frames = target_frames - len(adjusted_frames)
                frames_to_add = original_frames[
                    : min(remaining_frames, num_original_frames)
                ]
                adjusted_frames.extend(frames_to_add)

        elif mode == "pingpong":
            # Pingpong 模式：原始序列 + 反向序列 (去头尾)
            reversed_frames = original_frames[::-1][1:-1]  # 反向，去掉首尾避免重复
            pingpong_cycle = original_frames + reversed_frames

            # 循环 pingpong 序列直到达到目标帧数
            num_pingpong_frames = len(pingpong_cycle)
            while len(adjusted_frames) < target_frames:
                remaining_frames = target_frames - len(adjusted_frames)
                frames_to_add = pingpong_cycle[
                    : min(remaining_frames, num_pingpong_frames)
                ]
                adjusted_frames.extend(frames_to_add)

        # 将调整后的帧列表转换为张量
        adjusted_frames_tensor = torch.stack(adjusted_frames).cpu()

        # 确保返回的音频波形是 BATCH x Channels x Samples 格式
        padded_waveform = padded_waveform.unsqueeze(0).cpu()

        return (
            adjusted_frames_tensor,
            {"waveform": padded_waveform, "sample_rate": sample_rate},
        )


class SaveLipSyncVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "AUDIO": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_Video"}),
                "fps": (
                    "FLOAT",
                    {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.1},
                ),
                "quality": (
                    "INT",
                    {"default": 9, "min": 1, "max": 30, "step": 1},
                ),  # CRF for libx264/libx265 (lower is better quality, larger file)
                "codec": (
                    ["libx264", "libx265"],
                    {"default": "libx265"},
                ),
                "audio_bitrate": (
                    ["192k", "256k", "384k", "512k"],
                    {"default": "384k"},
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "merge_video"
    OUTPUT_NODE = True
    CATEGORY = "LatentSync"

    def merge_video(self, IMAGE, AUDIO, filename_prefix, fps, quality, codec, audio_bitrate):
        output_dir = folder_paths.get_output_directory()
        temp_dir = tempfile.mkdtemp(
            prefix="comfyui_imgio_temp_", dir=folder_paths.get_temp_directory()
        )
        silent_video_path = os.path.join(temp_dir, "silent_video.mp4")
        unique_id = str(uuid.uuid4())[:8]
        final_video_filename = f"{filename_prefix}_{unique_id}.mp4"
        final_video_path = os.path.join(output_dir, final_video_filename)
        temp_audio_path = None

        try:
            video_frames = []
            for i in range(IMAGE.shape[0]):
                img_np = IMAGE[i].cpu().numpy()
                img_uint8 = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                video_frames.append(img_uint8)

            with imageio.get_writer(
                silent_video_path,
                fps=fps,
                codec=codec,
                quality=None,
                pixelformat="rgb24",
                macro_block_size=None,
                ffmpeg_params=["-crf", str(quality)],
                ffmpeg_log_level="error",
            ) as writer:
                for frame in video_frames:
                    writer.append_data(frame)
        except Exception as e:
            print(f"Error during video writing: {e}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return {}

        audio_waveform = AUDIO.get("waveform", None)
        sample_rate = AUDIO.get("sample_rate", None)

        # 修改无音频情况下的返回
        if audio_waveform is None or sample_rate is None:
            try:
                shutil.copy(silent_video_path, final_video_path)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                return {
                    "ui": {
                        "videos": [
                            {
                                "filename": final_video_filename,
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                }

            except Exception as e:
                print(f"Error copying video: {e}")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                return {}

        try:
            temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
            audio_waveform = audio_waveform.cpu()
            if audio_waveform.ndim == 3:
                audio_waveform = audio_waveform.squeeze(0)
            elif audio_waveform.ndim == 1:
                audio_waveform = audio_waveform.unsqueeze(0)
            torchaudio.save(temp_audio_path, audio_waveform, sample_rate)
        except Exception as e:
            print(f"Error saving audio: {e}")
            shutil.copy(silent_video_path, final_video_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return {
                "ui": {
                    "videos": [
                        {
                            "filename": final_video_filename,
                            "subfolder": "",
                            "type": "output",
                        }
                    ]
                }
            }

        try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                silent_video_path,
                "-i",
                temp_audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                final_video_path,
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        except Exception as e:
            print(f"Error merging audio and video: {e}")
            shutil.copy(silent_video_path, final_video_path)
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        # 修改最终的返回
        return {
            "ui": {
                "videos": [
                    {
                        "filename": final_video_filename,
                        "subfolder": "",
                        "type": "output",
                    }
                ]
            }
        }
