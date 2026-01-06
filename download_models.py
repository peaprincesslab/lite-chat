#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本
用于下载模型到本地缓存
"""

import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import requests

# ==================== 模型配置常量 ====================
# 本地模型存储目录
LOCAL_MODELS_DIR = "./models"

# Stable Diffusion 模型配置
SD_MODEL_NAME = "radames/stable-diffusion-v1-5-img2img"
SD_LOCAL_DIR_NAME = "stable-diffusion-v1-5-img2img"

# BLIP 模型配置
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
BLIP_LOCAL_DIR_NAME = "blip-image-captioning-base"
BLIP_FILES = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json"
]

# Qwen 对话模型配置
QWEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct"
QWEN_LOCAL_DIR_NAME = "Qwen3-4B-Instruct"

# 镜像站点配置
MIRROR_SITES = [
    "https://hf-mirror.com",  # HuggingFace 镜像站
    "https://huggingface.co",  # 官方站点
]

# Stable Diffusion 核心文件列表
SD_CORE_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/pytorch_model.bin",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "unet/config.json",
    "unet/diffusion_pytorch_model.bin",
    "vae/config.json",
    "vae/diffusion_pytorch_model.bin",
    "feature_extractor/preprocessor_config.json"
]

# Stable Diffusion 完整文件列表（核心文件 + safetensors）
SD_FULL_FILES = SD_CORE_FILES + ["v1-5-pruned-emaonly.safetensors"]

def download_file_with_progress(url, filepath, filename):
    """带进度条的文件下载"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"下载 {filename} 失败: {e}")
        return False

def download_stable_diffusion_model():
    """下载Stable Diffusion模型"""
    # 创建models目录
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    
    print(f"正在下载模型: {SD_MODEL_NAME}")
    print(f"下载到目录: {LOCAL_MODELS_DIR}")
    
    # 询问用户选择下载版本
    print("\n选择要下载的模型版本:")
    print("1. 核心文件 (约2.2GB) - 仅包含diffusers格式文件")
    print("2. 核心文件 + v1-5-pruned-emaonly.safetensors (约6.5GB) - 推荐")
    print("3. 完整下载 (约12GB) - 包含所有文件")
    
    while True:
        choice = input("请选择 (1-3): ").strip()
        
        if choice == "1":
            # 只下载核心diffusers文件
            files_to_download = SD_CORE_FILES.copy()
            print("将下载核心文件 (约2.2GB)")
            break
            
        elif choice == "2":
            # 核心文件 + safetensors
            files_to_download = SD_FULL_FILES.copy()
            print("将下载核心文件 + safetensors (约6.5GB)")
            break
            
        elif choice == "3":
            # 完整下载
            files_to_download = None  # 下载所有文件
            print("将下载完整模型 (约12GB)")
            break
            
        else:
            print("无效选择，请输入1-3")
    
    print("这可能需要一些时间，请耐心等待...")
    
    for mirror in MIRROR_SITES:
        try:
            print(f"尝试使用镜像站点: {mirror}")
            
            # 设置环境变量使用镜像
            os.environ['HF_ENDPOINT'] = mirror
            
            if files_to_download is None:
                # 下载整个仓库
                cache_dir = snapshot_download(
                    repo_id=SD_MODEL_NAME,
                    cache_dir=LOCAL_MODELS_DIR,
                    local_files_only=False,
                    resume_download=True
                )
            else:
                # 只下载指定文件
                model_path = os.path.join(LOCAL_MODELS_DIR, SD_LOCAL_DIR_NAME)
                os.makedirs(model_path, exist_ok=True)
                
                print(f"开始下载 {len(files_to_download)} 个文件...")
                
                # 文件下载URL映射
                file_urls = {
                    file_path: f"{mirror}/{SD_MODEL_NAME}/resolve/main/{file_path}"
                    for file_path in files_to_download
                }
                
                success_count = 0
                for file_path in files_to_download:
                    if file_path in file_urls:
                        # 创建子目录
                        file_dir = os.path.dirname(os.path.join(model_path, file_path))
                        if file_dir:
                            os.makedirs(file_dir, exist_ok=True)
                        
                        file_full_path = os.path.join(model_path, file_path)
                        url = file_urls[file_path]
                        
                        print(f"\n正在下载: {file_path}")
                        if download_file_with_progress(url, file_full_path, file_path):
                            success_count += 1
                            print(f"✅ {file_path} 下载完成")
                        else:
                            print(f"❌ {file_path} 下载失败")
                    else:
                        print(f"⚠️  未找到 {file_path} 的下载链接")
                
                if success_count == len(files_to_download):
                    print(f"\n🎉 所有文件下载完成！({success_count}/{len(files_to_download)})")
                    cache_dir = model_path
                else:
                    print(f"\n⚠️  部分文件下载失败 ({success_count}/{len(files_to_download)})")
                    cache_dir = model_path
            
            print(f"模型下载成功！")
            print(f"使用镜像: {mirror}")
            print(f"本地目录: {cache_dir}")
            return True
            
        except Exception as e:
            print(f"镜像 {mirror} 下载失败: {e}")
            continue
    
    print("所有镜像站点都下载失败")
    return False

def download_blip_model():
    """下载BLIP模型（用于图生文）"""
    # 创建models目录
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    
    print(f"正在下载BLIP模型: {BLIP_MODEL_NAME}")
    print(f"下载到目录: {LOCAL_MODELS_DIR}")
    
    for mirror in MIRROR_SITES:
        try:
            print(f"尝试使用镜像站点: {mirror}")
            
            model_path = os.path.join(LOCAL_MODELS_DIR, BLIP_LOCAL_DIR_NAME)
            os.makedirs(model_path, exist_ok=True)
            
            print(f"开始下载 {len(BLIP_FILES)} 个BLIP文件...")
            
            # 文件下载URL映射
            file_urls = {
                file_path: f"{mirror}/{BLIP_MODEL_NAME}/resolve/main/{file_path}"
                for file_path in BLIP_FILES
            }
            
            success_count = 0
            for file_path in BLIP_FILES:
                if file_path in file_urls:
                    file_full_path = os.path.join(model_path, file_path)
                    url = file_urls[file_path]
                    
                    print(f"\n正在下载: {file_path}")
                    if download_file_with_progress(url, file_full_path, file_path):
                        success_count += 1
                        print(f"✅ {file_path} 下载完成")
                    else:
                        print(f"❌ {file_path} 下载失败")
                else:
                    print(f"⚠️  未找到 {file_path} 的下载链接")
            
            if success_count == len(BLIP_FILES):
                print(f"\n🎉 BLIP模型下载完成！({success_count}/{len(BLIP_FILES)})")
                print(f"使用镜像: {mirror}")
                print(f"本地目录: {model_path}")
                return True
            else:
                print(f"\n⚠️  BLIP模型部分文件下载失败 ({success_count}/{len(BLIP_FILES)})")
                continue
            
        except Exception as e:
            print(f"镜像 {mirror} 下载失败: {e}")
            continue
    
    print("所有镜像站点都下载失败")
    return False

def download_qwen_model():
    """下载Qwen对话模型（Qwen3-4B-Instruct）"""
    # 创建models目录
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    
    print(f"\n正在下载模型: {QWEN_MODEL_NAME}")
    print(f"下载到目录: {LOCAL_MODELS_DIR}")
    print("这可能需要一些时间，请耐心等待...")
    
    for mirror in MIRROR_SITES:
        try:
            print(f"\n尝试使用镜像站点: {mirror}")
            
            # 设置环境变量使用镜像
            os.environ['HF_ENDPOINT'] = mirror
            
            # 使用snapshot_download下载整个模型仓库
            model_path = os.path.join(LOCAL_MODELS_DIR, QWEN_LOCAL_DIR_NAME)
            
            print("开始下载模型文件...")
            print("注意：模型文件较大，下载可能需要较长时间")
            
            cache_dir = snapshot_download(
                repo_id=QWEN_MODEL_NAME,
                cache_dir=model_path,
                local_files_only=False,
                resume_download=True
            )
            
            print(f"\n🎉 Qwen模型下载完成！")
            print(f"使用镜像: {mirror}")
            print(f"本地目录: {cache_dir}")
            print(f"\n模型已保存到: {model_path}")
            print("您可以在代码中使用以下路径加载模型:")
            print(f'  model_path = "{model_path}"')
            return True
            
        except Exception as e:
            print(f"镜像 {mirror} 下载失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("所有镜像站点都下载失败")
    return False

def main():
    """主函数"""
    print("=" * 50)
    print("Hugging Face 模型下载工具")
    print("=" * 50)
    
    # 检查网络连接（使用镜像站点）
    network_ok = False
    for mirror in MIRROR_SITES:
        try:
            import requests
            response = requests.get(mirror, timeout=10)
            print(f"网络连接正常 - 使用镜像: {mirror}")
            network_ok = True
            # 设置默认镜像
            os.environ['HF_ENDPOINT'] = mirror
            break
        except Exception as e:
            print(f"镜像 {mirror} 连接失败: {e}")
            continue
    
    if not network_ok:
        print("所有镜像站点都无法连接，请检查网络连接")
        return
    
    print("\n选择要下载的模型:")
    print("1. Stable Diffusion v1.5 (文生图/图生图) - 约6.5GB")
    print("2. BLIP (图生文) - 约1GB")
    print("3. Qwen对话模型 (Qwen3-4B-Instruct) - 约8GB")
    print("4. 全部下载")
    print("5. 退出")
    
    while True:
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            success = download_stable_diffusion_model()
            if success:
                print("✅ Stable Diffusion模型下载完成")
            else:
                print("❌ Stable Diffusion模型下载失败")
            break
            
        elif choice == "2":
            success = download_blip_model()
            if success:
                print("✅ BLIP模型下载完成")
            else:
                print("❌ BLIP模型下载失败")
            break
            
        elif choice == "3":
            success = download_qwen_model()
            if success:
                print("✅ Qwen模型下载完成")
            else:
                print("❌ Qwen模型下载失败")
            break
            
        elif choice == "4":
            print("开始下载所有模型...")
            sd_success = download_stable_diffusion_model()
            blip_success = download_blip_model()
            qwen_success = download_qwen_model()
            
            if sd_success and blip_success and qwen_success:
                print("✅ 所有模型下载完成")
            else:
                print("❌ 部分模型下载失败")
            break
            
        elif choice == "5":
            print("退出")
            break
            
        else:
            print("无效选择，请输入1-5")

if __name__ == "__main__":
    main()
