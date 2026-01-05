#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载模块
统一管理所有模型的加载，实现懒加载机制
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.trainer_utils import set_seed
import random

# ==================== 配置 ====================
# 默认模型路径（使用项目下的models目录）
DEFAULT_MODELS_DIR = "./models"
DEFAULT_CHAT_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, "Qwen3-4B-Instruct")
DEFAULT_IMAGE_MODEL_NAME = "radames/stable-diffusion-v1-5-img2img"
DEFAULT_CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cpu_only = not torch.cuda.is_available()

# 镜像站点配置
HF_MIRROR = "https://hf-mirror.com"
OFFLINE_MODE = True  # 是否只使用本地缓存的模型

# 默认生成参数
DEFAULT_MAX_NEW_TOKENS = 8192

# ==================== 模型缓存 ====================
# 使用字典存储已加载的模型，实现懒加载
_model_cache = {
    'chat_model': None,
    'chat_tokenizer': None,
    'image_generator': None,
    'img2img_generator': None,
    'caption_processor': None,
    'caption_model': None,
}

# ==================== 对话模型加载 ====================
def load_chat_model(model_path=None, cpu_only=None):
    """
    加载对话模型（懒加载）
    
    Args:
        model_path: 模型路径，默认为 ./models/Qwen3-4B-Instruct
        cpu_only: 是否只使用CPU，默认自动检测
    
    Returns:
        (model, tokenizer): 模型和分词器
    """
    # 如果已加载，直接返回
    if _model_cache['chat_model'] is not None and _model_cache['chat_tokenizer'] is not None:
        return _model_cache['chat_model'], _model_cache['chat_tokenizer']
    
    # 使用默认路径
    if model_path is None:
        model_path = DEFAULT_CHAT_MODEL_PATH
    
    if cpu_only is None:
        cpu_only = not torch.cuda.is_available()
    
    print(f"正在加载对话模型，模型路径: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型路径不存在: {model_path}\n"
            f"请检查路径是否正确，或确保模型已下载到该位置。\n"
            f"可以使用以下方式下载模型：\n"
            f"1. 使用下载脚本: python download_models.py\n"
            f"2. 手动下载到 {model_path}"
        )
    
    print(f"模型路径验证通过: {model_path}")
    
    # 加载分词器
    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, resume_download=True)
    print("分词器加载完成")
    
    # 确保pad_token设置正确
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("设置pad_token为eos_token")
    
    # 加载模型
    device_map = "cpu" if cpu_only else "auto"
    print(f"正在加载模型，设备映射: {device_map}，精度: {'float16' if not cpu_only else 'float32'}")
    
    # 检查模型文件大小
    try:
        total_size = 0
        model_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    total_size += size
                    if file.endswith(('.bin', '.safetensors', '.pt', '.pth')):
                        model_files.append((file, size / 1024**3))  # GB
        
        if total_size > 0:
            print(f"模型总大小: {total_size / 1024**3:.2f} GB")
            if model_files:
                print("主要模型文件:")
                for file, size in model_files[:5]:  # 只显示前5个
                    print(f"  - {file}: {size:.2f} GB")
            print(f"预计加载时间: {total_size / 1024**3 / 2:.1f} - {total_size / 1024**3:.1f} 分钟（取决于磁盘速度）")
    except Exception as e:
        print(f"无法计算模型大小: {e}")
    
    # 显示显存信息
    if not cpu_only and torch.cuda.is_available():
        print(f"GPU 显存信息: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"当前显存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    print("\n开始加载模型权重...")
    print("这可能需要几分钟时间，请耐心等待...")
    print("如果长时间（超过10分钟）没有响应，可能是显存不足或模型文件有问题。")
    
    try:
        # 使用 low_cpu_mem_usage 来减少内存使用
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if not cpu_only else torch.float32,
            device_map=device_map,
            resume_download=True,
            low_cpu_mem_usage=True,
        ).eval()
        print("模型加载完成，正在优化...")
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"\n错误: 显存不足！")
            print(f"错误详情: {str(e)}")
            print("\n建议:")
            print("1. 关闭其他占用显存的程序")
            print("2. 尝试使用 CPU 模式")
            print("3. 使用更小的模型或减少 batch_size")
        raise
    except Exception as e:
        print(f"\n模型加载失败: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise
    
    # 如果使用 GPU，确保模型使用半精度以节省显存
    if not cpu_only and torch.cuda.is_available():
        try:
            model.half()
            print("模型已切换为半精度（float16）。")
        except:
            print("无法切换模型为半精度，继续使用默认精度。")
    
    model.generation_config.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    print("模型优化完成")
    
    # 缓存模型
    _model_cache['chat_model'] = model
    _model_cache['chat_tokenizer'] = tokenizer
    
    return model, tokenizer

# ==================== 图像生成模型加载 ====================
def load_image_generator():
    """
    加载文生图模型（懒加载）
    
    Returns:
        pipe: StableDiffusionPipeline 对象
    """
    # 如果已加载，直接返回
    if _model_cache['image_generator'] is not None:
        return _model_cache['image_generator']
    
    print("正在加载图像生成模型...")
    
    # 设置镜像站点
    os.environ['HF_ENDPOINT'] = HF_MIRROR
    
    # 检查本地models目录
    local_model_path = os.path.join(DEFAULT_MODELS_DIR, "stable-diffusion-v1-5-img2img")
    
    try:
        # 根据配置选择加载方式
        if OFFLINE_MODE:
            # 离线模式：优先使用本地models目录，然后尝试HuggingFace缓存
            if os.path.exists(local_model_path):
                print(f"从本地models目录加载模型: {local_model_path}")
                pipe = StableDiffusionPipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16 if not cpu_only else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True
                )
                print("从本地models目录加载图像生成模型成功")
            else:
                print("本地models目录未找到模型，尝试HuggingFace缓存...")
                pipe = StableDiffusionPipeline.from_pretrained(
                    DEFAULT_IMAGE_MODEL_NAME,
                    torch_dtype=torch.float16 if not cpu_only else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True
                )
                print("从HuggingFace缓存加载图像生成模型成功")
        else:
            # 在线模式：尝试本地models目录，然后HuggingFace缓存，最后网络下载
            if os.path.exists(local_model_path):
                print(f"从本地models目录加载模型: {local_model_path}")
                pipe = StableDiffusionPipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16 if not cpu_only else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True
                )
                print("从本地models目录加载图像生成模型成功")
            else:
                try:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        DEFAULT_IMAGE_MODEL_NAME,
                        torch_dtype=torch.float16 if not cpu_only else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        local_files_only=True
                    )
                    print("从HuggingFace缓存加载图像生成模型成功")
                except Exception as local_error:
                    print(f"本地缓存加载失败: {local_error}")
                    print("尝试从网络下载模型...")
                    pipe = StableDiffusionPipeline.from_pretrained(
                        DEFAULT_IMAGE_MODEL_NAME,
                        torch_dtype=torch.float16 if not cpu_only else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        local_files_only=False
                    )
                    print("从网络下载图像生成模型成功")
        
        if not cpu_only and torch.cuda.is_available():
            pipe = pipe.to("cuda")
            print("图像生成模型已加载到GPU")
        else:
            print("图像生成模型已加载到CPU")
        
        # 缓存模型
        _model_cache['image_generator'] = pipe
        return pipe
        
    except Exception as e:
        print(f"加载图像生成模型失败: {e}")
        print("提示：请确保模型已下载到本地")
        print("可以使用以下方式下载模型：")
        print("1. 使用提供的下载脚本: python download_models.py")
        print("2. 手动下载到models目录")
        print("3. 使用huggingface-cli: huggingface-cli download runwayml/stable-diffusion-v1-5")
        return None

# ==================== 图生图模型加载 ====================
def load_img2img_generator():
    """
    加载图生图模型（懒加载）
    
    Returns:
        pipe: StableDiffusionImg2ImgPipeline 对象
    """
    # 如果已加载，直接返回
    if _model_cache['img2img_generator'] is not None:
        return _model_cache['img2img_generator']
    
    print("正在加载图生图模型...")
    
    # 设置镜像站点
    os.environ['HF_ENDPOINT'] = HF_MIRROR
    
    # 检查本地models目录
    local_model_path = os.path.join(DEFAULT_MODELS_DIR, "stable-diffusion-v1-5-img2img")
    
    try:
        # 根据配置选择加载方式
        if OFFLINE_MODE:
            # 离线模式：优先使用本地models目录，然后尝试HuggingFace缓存
            if os.path.exists(local_model_path):
                print(f"从本地models目录加载模型: {local_model_path}")
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16 if not cpu_only else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True
                )
                print("从本地models目录加载图生图模型成功")
            else:
                print("本地models目录未找到模型，尝试HuggingFace缓存...")
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    DEFAULT_IMAGE_MODEL_NAME,
                    torch_dtype=torch.float16 if not cpu_only else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True
                )
                print("从HuggingFace缓存加载图生图模型成功")
        else:
            # 在线模式：尝试本地models目录，然后HuggingFace缓存，最后网络下载
            if os.path.exists(local_model_path):
                print(f"从本地models目录加载模型: {local_model_path}")
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16 if not cpu_only else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True
                )
                print("从本地models目录加载图生图模型成功")
            else:
                try:
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        DEFAULT_IMAGE_MODEL_NAME,
                        torch_dtype=torch.float16 if not cpu_only else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        local_files_only=True
                    )
                    print("从HuggingFace缓存加载图生图模型成功")
                except Exception as local_error:
                    print(f"本地缓存加载失败: {local_error}")
                    print("尝试从网络下载模型...")
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        DEFAULT_IMAGE_MODEL_NAME,
                        torch_dtype=torch.float16 if not cpu_only else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        local_files_only=False
                    )
                    print("从网络下载图生图模型成功")
        
        if not cpu_only and torch.cuda.is_available():
            pipe = pipe.to("cuda")
            print("图生图模型已加载到GPU")
        else:
            print("图生图模型已加载到CPU")
        
        # 缓存模型
        _model_cache['img2img_generator'] = pipe
        return pipe
        
    except Exception as e:
        print(f"加载图生图模型失败: {e}")
        print("提示：请确保模型已下载到本地")
        print("可以使用以下方式下载模型：")
        print("1. 使用提供的下载脚本: python download_models.py")
        print("2. 手动下载到models目录")
        print("3. 使用huggingface-cli: huggingface-cli download runwayml/stable-diffusion-v1-5")
        return None

# ==================== 图像描述模型加载 ====================
def load_caption_model():
    """
    加载图像描述生成模型（懒加载）
    
    Returns:
        (processor, model): BlipProcessor 和 BlipForConditionalGeneration 对象
    """
    # 如果已加载，直接返回
    if _model_cache['caption_processor'] is not None and _model_cache['caption_model'] is not None:
        return _model_cache['caption_processor'], _model_cache['caption_model']
    
    print("正在加载图像描述生成模型...")
    
    # 设置镜像站点
    os.environ['HF_ENDPOINT'] = HF_MIRROR
    
    # 检查本地models目录
    local_model_path = os.path.join(DEFAULT_MODELS_DIR, "blip-image-captioning-base")
    
    try:
        # 根据配置选择加载方式
        if OFFLINE_MODE:
            # 离线模式：优先使用本地models目录，然后尝试HuggingFace缓存
            if os.path.exists(local_model_path):
                print(f"从本地models目录加载模型: {local_model_path}")
                processor = BlipProcessor.from_pretrained(
                    local_model_path,
                    local_files_only=True
                )
                model = BlipForConditionalGeneration.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16 if not cpu_only else torch.float32,
                    local_files_only=True
                )
                print("从本地models目录加载图像描述模型成功")
            else:
                print("本地models目录未找到模型，尝试HuggingFace缓存...")
                processor = BlipProcessor.from_pretrained(
                    DEFAULT_CAPTION_MODEL_NAME,
                    local_files_only=True
                )
                model = BlipForConditionalGeneration.from_pretrained(
                    DEFAULT_CAPTION_MODEL_NAME,
                    torch_dtype=torch.float16 if not cpu_only else torch.float32,
                    local_files_only=True
                )
                print("从HuggingFace缓存加载图像描述模型成功")
        else:
            # 在线模式：尝试本地models目录，然后HuggingFace缓存，最后网络下载
            if os.path.exists(local_model_path):
                print(f"从本地models目录加载模型: {local_model_path}")
                processor = BlipProcessor.from_pretrained(
                    local_model_path,
                    local_files_only=True
                )
                model = BlipForConditionalGeneration.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16 if not cpu_only else torch.float32,
                    local_files_only=True
                )
                print("从本地models目录加载图像描述模型成功")
            else:
                try:
                    processor = BlipProcessor.from_pretrained(
                        DEFAULT_CAPTION_MODEL_NAME,
                        local_files_only=True
                    )
                    model = BlipForConditionalGeneration.from_pretrained(
                        DEFAULT_CAPTION_MODEL_NAME,
                        torch_dtype=torch.float16 if not cpu_only else torch.float32,
                        local_files_only=True
                    )
                    print("从HuggingFace缓存加载图像描述模型成功")
                except Exception as local_error:
                    print(f"本地缓存加载失败: {local_error}")
                    print("尝试从网络下载模型...")
                    processor = BlipProcessor.from_pretrained(
                        DEFAULT_CAPTION_MODEL_NAME,
                        local_files_only=False
                    )
                    model = BlipForConditionalGeneration.from_pretrained(
                        DEFAULT_CAPTION_MODEL_NAME,
                        torch_dtype=torch.float16 if not cpu_only else torch.float32,
                        local_files_only=False
                    )
                    print("从网络下载图像描述模型成功")
        
        if not cpu_only and torch.cuda.is_available():
            model = model.to("cuda")
            print("图像描述模型已加载到GPU")
        else:
            print("图像描述模型已加载到CPU")
        
        # 缓存模型
        _model_cache['caption_processor'] = processor
        _model_cache['caption_model'] = model
        
        return processor, model
        
    except Exception as e:
        print(f"加载图像描述模型失败: {e}")
        print("提示：请确保模型已下载到本地")
        print("可以使用以下方式下载模型：")
        print("1. 使用提供的下载脚本: python download_models.py")
        print("2. 手动下载到models目录")
        print("3. 使用huggingface-cli: huggingface-cli download Salesforce/blip-image-captioning-base")
        return None, None

# ==================== 工具函数 ====================
def clear_model_cache():
    """清空所有模型缓存"""
    global _model_cache
    _model_cache = {
        'chat_model': None,
        'chat_tokenizer': None,
        'image_generator': None,
        'img2img_generator': None,
        'caption_processor': None,
        'caption_model': None,
    }
    # 清理GPU显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("模型缓存已清空")

def get_model_cache_status():
    """获取模型缓存状态"""
    status = {}
    for key, value in _model_cache.items():
        status[key] = value is not None
    return status
