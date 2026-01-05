import sys
import os

import torch
from transformers import TextIteratorStreamer
from transformers.trainer_utils import set_seed
from threading import Thread
import random
import gradio as gr
import re
from PIL import Image
import io
import base64
import numpy as np

# 导入模型加载模块
from load_models import (
    load_chat_model,
    load_image_generator,
    load_img2img_generator,
    load_caption_model,
    DEVICE,
    cpu_only,
    DEFAULT_MAX_NEW_TOKENS
)

# 文件处理相关函数
def read_file_content(file_path):
    """读取文件内容"""
    try:
        # 获取文件扩展名
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
            # 文本文件，尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            content = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                return f"无法读取文件，尝试了多种编码都失败"
        elif file_ext in ['.docx']:
            # Word文档
            try:
                from docx import Document
                doc = Document(file_path)
                content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                return f"无法读取Word文档，请安装python-docx: pip install python-docx"
        elif file_ext in ['.pdf']:
            # PDF文档
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ''
                    for page in pdf_reader.pages:
                        content += page.extract_text() + '\n'
            except ImportError:
                return f"无法读取PDF文档，请安装PyPDF2: pip install PyPDF2"
            except Exception as e:
                return f"读取PDF文件时出错: {str(e)}"
        else:
            return f"不支持的文件格式: {file_ext}"
        
        return content.strip()
    except Exception as e:
        return f"读取文件时出错: {str(e)}"

def process_file_content(content, max_length=2000):
    """处理文件内容，限制长度并格式化"""
    if not content:
        return ""
    
    # 移除多余的空白字符
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = content.strip()
    
    # 如果内容太长，截取前max_length个字符
    if len(content) > max_length:
        content = content[:max_length] + "\n\n[内容已截断，仅显示前{}个字符]".format(max_length)
    
    return content

def format_file_context(file_content, file_name):
    """格式化文件内容为上下文"""
    if not file_content:
        return ""
    
    return f"文件《{file_name}》的内容：\n\n{file_content}\n\n请基于以上文件内容进行创作或回答相关问题。"

# 图像生成相关函数

def generate_image(prompt, negative_prompt="", num_inference_steps=20, guidance_scale=7.5, width=512, height=512):
    """生成图像"""
    if not hasattr(generate_image, 'pipe') or generate_image.pipe is None:
        generate_image.pipe = load_image_generator()
    
    if generate_image.pipe is None:
        return None, "图像生成模型未加载，请切换到图像生成页面后重试"
    
    try:
        # 生成图像
        result = generate_image.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        )
        
        return result.images[0], None
    except Exception as e:
        return None, f"图像生成失败: {str(e)}"

def save_image_to_base64(image):
    """将图像转换为base64字符串"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        return None


def generate_img2img(prompt, init_image, negative_prompt="", num_inference_steps=20, guidance_scale=7.5, strength=0.75, width=512, height=512):
    """图生图功能"""
    if not hasattr(generate_img2img, 'pipe') or generate_img2img.pipe is None:
        generate_img2img.pipe = load_img2img_generator()
    
    if generate_img2img.pipe is None:
        return None, "图生图模型未加载，请切换到图生图页面后重试"
    
    try:
        # 确保输入图像是PIL Image格式
        if isinstance(init_image, str):
            init_image = Image.open(init_image)
        elif isinstance(init_image, np.ndarray):
            init_image = Image.fromarray(init_image)
        
        # 调整图像尺寸到目标分辨率
        init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)
        
        # 生成图像
        result = generate_img2img.pipe(
            prompt=prompt,
            image=init_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength
        )
        
        return result.images[0], None
    except Exception as e:
        return None, f"图生图生成失败: {str(e)}"

def preprocess_image(image, target_size=(512, 512)):
    """预处理输入图像"""
    try:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 调整尺寸
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        return None, f"图像预处理失败: {str(e)}"


def generate_image_caption(image, max_length=100):
    """生成图像描述"""
    if not hasattr(generate_image_caption, 'processor') or generate_image_caption.processor is None:
        generate_image_caption.processor, generate_image_caption.model = load_caption_model()
    
    if generate_image_caption.processor is None or generate_image_caption.model is None:
        return None, "图像描述模型未加载，请切换到图生文页面后重试"
    
    try:
        # 确保输入图像是PIL Image格式
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 处理图像
        inputs = generate_image_caption.processor(image, return_tensors="pt")
        
        # 移动到设备
        if not cpu_only and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 生成描述
        out = generate_image_caption.model.generate(
            **inputs, 
            max_length=max_length,
            num_beams=3,
            early_stopping=True,
            do_sample=True,
            temperature=0.7
        )
        caption = generate_image_caption.processor.decode(out[0], skip_special_tokens=True)
        
        # 直接返回生成的描述，不做中文转换
        # 中文转换将在改写阶段由其他模型处理
        
        return caption, None
    except Exception as e:
        return None, f"图像描述生成失败: {str(e)}"

def rewrite_caption_with_prompt(caption, rewrite_prompt):
    """使用提示词改写图像描述"""
    global model, tokenizer
    
    # 懒加载：如果模型未加载，则加载模型
    if model is None or tokenizer is None:
        print("首次使用对话功能，正在加载对话模型...")
        try:
            model, tokenizer = load_chat_model()
            print("对话模型加载完成！")
        except Exception as e:
            return None, f"模型加载失败: {str(e)}\n请检查模型路径是否正确，或使用 download_models.py 下载模型。"
    
    try:
        # 构建改写提示
        system_prompt = "你是一个专业的图像描述改写助手。请根据用户的要求改写图像描述，保持描述的准确性和流畅性。"
        user_prompt = f"原始图像描述：{caption}\n\n改写要求：{rewrite_prompt}\n\n请提供改写后的图像描述："
        
        # 构建对话
        conversation = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        # 应用聊天模板
        try:
            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
        except AttributeError:
            text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        
        # Tokenize输入
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
        # 生成改写后的描述
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        rewritten_caption = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return rewritten_caption.strip(), None
    except Exception as e:
        return None, f"描述改写失败: {str(e)}"

# 默认参数
DEFAULT_TOP_P = 0.9        # Top-p (nucleus sampling) 范围在0到1之间
DEFAULT_TOP_K = 80         # Top-k 采样的K值
DEFAULT_TEMPERATURE = 0.3  # 温度参数，控制生成文本的随机性
DEFAULT_REPETITION_PENALTY = 1.1  # 重复惩罚参数
DEFAULT_SYSTEM_MESSAGE = ""  # 默认系统消息

# 图像生成相关配置
DEFAULT_IMAGE_SIZE = (512, 512)
DEFAULT_NUM_INFERENCE_STEPS = 50  # 提高默认推理步数
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_STRENGTH = 0.6  # 降低默认强度，更好地保留原图特征

# 图像描述生成相关配置
DEFAULT_MAX_CAPTION_LENGTH = 100  # 最大描述长度

def _chat_stream(model, tokenizer, query, history, system_message, top_p, top_k, temperature, max_new_tokens, repetition_penalty, file_context=""):
    # 懒加载：如果模型未加载，则加载模型
    if model is None or tokenizer is None:
        print("首次使用对话功能，正在加载对话模型...")
        try:
            model, tokenizer = load_chat_model()
            print("对话模型加载完成！")
        except Exception as e:
            yield f"模型加载失败: {str(e)}\n请检查模型路径是否正确，或使用 download_models.py 下载模型。"
            return
    
    # 检查是否包含图像生成命令
    if "/generate_image" in query or "/画图" in query or "/生成图像" in query:
        # 提取图像描述
        if "/generate_image" in query:
            prompt = query.replace("/generate_image", "").strip()
        elif "/画图" in query:
            prompt = query.replace("/画图", "").strip()
        elif "/生成图像" in query:
            prompt = query.replace("/生成图像", "").strip()
        else:
            prompt = query
        
        if not prompt:
            yield "请提供图像描述，例如：/画图 一只可爱的小猫"
            return
        
        # 生成图像
        image, error = generate_image(prompt)
        if error:
            yield f"图像生成失败: {error}"
            return
        
        # 保存图像并返回base64
        image_b64 = save_image_to_base64(image)
        if image_b64:
            yield f"已为您生成图像：\n![生成的图像]({image_b64})\n\n描述：{prompt}"
        else:
            yield f"图像生成成功，但无法显示。描述：{prompt}"
        return
    
    # 检查是否包含图生图命令
    if "/img2img" in query or "/图生图" in query or "/基于图像生成" in query:
        yield "图生图功能需要在图像生成界面中使用。请切换到'图像生成'标签页，然后选择'图生图'子标签页，上传参考图像并输入描述。"
        return
    
    # 检查是否包含图生文命令
    if "/img2text" in query or "/图生文" in query or "/图像描述" in query or "/描述图像" in query:
        yield "图生文功能需要在图像生成界面中使用。请切换到'图像生成'标签页，然后选择'图生文'子标签页，上传图像即可生成描述，还可以使用提示词改写描述。"
        return
    
    # 如果有文件上下文，将其添加到系统消息中
    enhanced_system_message = system_message
    if file_context:
        enhanced_system_message = f"{system_message}\n\n{file_context}" if system_message else file_context
    
    # 添加图像生成功能说明到系统消息
    image_help = "\n\n注意：您可以使用以下命令生成图像：\n- /画图 [描述] - 生成图像\n- /generate_image [描述] - 生成图像\n- /生成图像 [描述] - 生成图像\n\n图生图功能：\n- /图生图 - 查看图生图使用说明\n- /img2img - 查看图生图使用说明\n- /基于图像生成 - 查看图生图使用说明\n\n图生文功能：\n- /图生文 - 查看图生文使用说明\n- /img2text - 查看图生文使用说明\n- /图像描述 - 查看图生文使用说明\n- /描述图像 - 查看图生文使用说明"
    enhanced_system_message = enhanced_system_message + image_help
    
    conversation = [
        {'role': 'system', 'content': enhanced_system_message},
    ]
    for query_h, response_h in history:
        conversation.append({'role': 'user', 'content': query_h})
        conversation.append({'role': 'assistant', 'content': response_h})
    conversation.append({'role': 'user', 'content': query})
    
    # 准备输入
    try:
        # 尝试使用 apply_chat_template 方法
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,               # 确保返回的是字符串
            add_generation_prompt=True
        )
    except AttributeError:
        # 如果没有 apply_chat_template 方法，使用标准方法构建对话
        print("[WARNING] `apply_chat_template` 方法不存在，使用标准对话格式。")
        text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation])
        text += "\nAssistant:"
    
    # 确保 text 是字符串
    if not isinstance(text, str):
        raise ValueError("apply_chat_template 应返回字符串类型的文本。")
    
    # Tokenize 输入
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    
    # 生成参数
    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # 添加attention mask
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        do_sample=True,  # 确保使用采样方法
        pad_token_id=tokenizer.eos_token_id,  # 避免警告
        eos_token_id=tokenizer.eos_token_id,  # 结束标记
        repetition_penalty=repetition_penalty,  # 重复惩罚，避免重复内容
        streamer=streamer,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield new_text
    return generated_text

def initialize_model():
    """初始化对话模型（懒加载，实际在首次使用时加载）"""
    seed = random.randint(0, 2**32 - 1)  # 随机生成一个种子
    set_seed(seed)  # 设置随机种子
    # 模型将在首次使用时通过 load_chat_model 加载
    return None, None

# 初始化模型和分词器（懒加载，实际在首次使用时加载）
print("模型将在首次使用时加载（懒加载模式）...")
model, tokenizer = initialize_model()

def chat_interface(user_input, history, system_message, top_p, top_k, temperature, max_new_tokens, repetition_penalty, file_context=""):
    global model, tokenizer
    
    if user_input.strip() == "":
        yield history, history, system_message, "", file_context
        return
    
    # 懒加载：如果模型未加载，则加载模型
    if model is None or tokenizer is None:
        print("首次使用对话功能，正在加载对话模型...")
        try:
            model, tokenizer = load_chat_model()
            print("对话模型加载完成！")
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}\n请检查模型路径是否正确，或使用 download_models.py 下载模型。"
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": error_msg})
            yield history, history, system_message, "", file_context
            return
    
    # 添加用户消息
    history.append({"role": "user", "content": user_input})
    # 添加空的助手消息
    history.append({"role": "assistant", "content": ""})
    yield history, history, system_message, "", file_context  # 更新 Chatbot 组件和状态

    # 获取模型生成的回复
    # 转换历史格式用于模型处理
    model_history = []
    for msg in history[:-2]:  # 排除最后两条消息
        if msg["role"] == "user":
            model_history.append((msg["content"], ""))
        elif msg["role"] == "assistant":
            if model_history:
                model_history[-1] = (model_history[-1][0], msg["content"])
    
    generator = _chat_stream(model, tokenizer, user_input, model_history, system_message, top_p, top_k, temperature, max_new_tokens, repetition_penalty, file_context)
    assistant_reply = ""
    for new_text in generator:
        assistant_reply += new_text
        # 更新最后一条助手回复
        updated_history = history.copy()
        updated_history[-1] = {"role": "assistant", "content": assistant_reply}
        yield updated_history, updated_history, system_message, "", file_context  # 更新 Chatbot 组件和状态

def clear_history():
    return [], [], DEFAULT_SYSTEM_MESSAGE, "", "", DEFAULT_REPETITION_PENALTY

def handle_file_upload(file):
    """处理文件上传"""
    if file is None:
        return "", ""  # 返回两个空字符串
    
    try:
        # 获取文件路径（Gradio可能返回文件对象或文件路径字符串）
        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name if hasattr(file, 'name') else str(file)
        
        # 读取文件内容
        content = read_file_content(file_path)
        
        # 检查是否是错误信息（包括所有可能的错误前缀）
        if (content.startswith("无法读取") or 
            content.startswith("读取文件时出错") or 
            content.startswith("不支持的文件格式") or
            content.startswith("处理文件时出错")):
            return content, ""  # 返回错误信息和空的文件上下文
        
        # 处理文件内容
        processed_content = process_file_content(content)
        file_name = os.path.basename(file_path)
        
        # 格式化文件上下文
        file_context = format_file_context(processed_content, file_name)
        
        # 返回文件预览和文件上下文
        return content, file_context
    except Exception as e:
        error_msg = f"处理文件时出错: {str(e)}"
        return error_msg, ""  # 返回错误信息和空的文件上下文

def handle_image_generation(prompt, negative_prompt, width, height, num_steps, guidance):
    """处理图像生成"""
    if not prompt.strip():
        return None, "请输入图像描述"
    
    try:
        # 生成图像
        image, error = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            width=width,
            height=height
        )
        
        if error:
            return None, error
        
        return image, "图像生成成功！"
    except Exception as e:
        return None, f"图像生成失败: {str(e)}"

def save_generated_image(image):
    """保存生成的图像"""
    if image is None:
        return "没有图像可保存"
    
    try:
        # 创建保存目录
        save_dir = "generated_images"
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        import time
        timestamp = int(time.time())
        filename = f"generated_image_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        
        # 保存图像
        image.save(filepath)
        return f"图像已保存到: {filepath}"
    except Exception as e:
        return f"保存图像失败: {str(e)}"

def handle_img2img_generation(prompt, init_image, negative_prompt, num_steps, guidance, strength, width, height):
    """处理图生图生成"""
    if not prompt.strip():
        return None, "请输入图像描述"
    
    if init_image is None:
        return None, "请上传参考图像"
    
    try:
        # 预处理输入图像
        processed_image = preprocess_image(init_image)
        if processed_image is None:
            return None, "图像预处理失败"
        
        # 生成图像
        image, error = generate_img2img(
            prompt=prompt,
            init_image=processed_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            strength=strength,
            width=width,
            height=height
        )
        
        if error:
            return None, error
        
        return image, "图生图生成成功！"
    except Exception as e:
        return None, f"图生图生成失败: {str(e)}"

def handle_image_caption_generation(image, max_length):
    """处理图像描述生成"""
    if image is None:
        return "", "请上传图像"
    
    try:
        # 生成图像描述
        caption, error = generate_image_caption(image, max_length)
        if error:
            return "", error
        
        return caption, "图像描述生成成功！"
    except Exception as e:
        return "", f"图像描述生成失败: {str(e)}"

def handle_caption_rewrite(caption, rewrite_prompt):
    """处理图像描述改写"""
    if not caption.strip():
        return "", "请先生成图像描述"
    
    if not rewrite_prompt.strip():
        return "", "请输入改写要求"
    
    try:
        # 使用主模型改写描述
        rewritten_caption, error = rewrite_caption_with_prompt(caption, rewrite_prompt)
        if error:
            return "", error
        
        return rewritten_caption, "描述改写成功！"
    except Exception as e:
        return "", f"描述改写失败: {str(e)}"

# Gradio 接口
with gr.Blocks() as demo:
    # CSS
    gr.HTML("""
    <style>
        #chat-container {
            height: 500px;
            overflow-y: auto;
        }
        .settings-column {
            padding-left: 20px;
            border-left: 1px solid #ddd;
        }
        .send-button {
            margin-top: 10px;
            width: 100%;
        }
    </style>
    """)

    gr.Markdown("# Qwen2.5 Sex")

    # 创建标签页
    with gr.Tabs() as tabs:
        # 文本对话标签页
        with gr.TabItem("文本对话"):
            with gr.Row():
                # 左侧：聊天区域
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(elem_id="chat-container", type="messages")
                    
                    # 文件上传区域
                    with gr.Row():
                        file_upload = gr.File(
                            label="上传文件",
                            file_types=['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.docx', '.pdf'],
                            file_count="single"
                        )
                        file_clear_btn = gr.Button("清除文件", size="sm")
                    
                    # 文件内容预览
                    file_preview = gr.Textbox(
                        label="文件内容预览",
                        placeholder="文件内容将在这里显示...",
                        lines=4,
                        interactive=False
                    )
                    
                    # 用户输入区域
                    user_input = gr.Textbox(
                        show_label=False, 
                        placeholder="输入你的问题...", 
                        lines=2,
                        interactive=True
                    )
                    send_btn = gr.Button("发送", elem_classes=["send-button"])
                
                # 右侧：参数设置
                with gr.Column(scale=1):
                    gr.Markdown("### 对话设置")
                    
                    # 系统消息
                    gr.Markdown("#### 系统消息")
                    system_message = gr.Textbox(
                        label="系统消息",
                        value=DEFAULT_SYSTEM_MESSAGE,
                        placeholder="输入系统消息...",
                        lines=3
                    )
                    
                    # 生成参数
                    gr.Markdown("#### 生成参数")
                    top_p_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=DEFAULT_TOP_P, step=0.05,
                        label="Top-p (nucleus sampling)",
                        info="控制词汇选择的随机性"
                    )
                    top_k_slider = gr.Slider(
                        minimum=0, maximum=100, value=DEFAULT_TOP_K, step=1,
                        label="Top-k",
                        info="限制候选词汇数量"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1, maximum=1.5, value=DEFAULT_TEMPERATURE, step=0.05,
                        label="Temperature",
                        info="控制生成的随机性"
                    )
                    max_new_tokens_slider = gr.Slider(
                        minimum=50, maximum=16384, value=DEFAULT_MAX_NEW_TOKENS, step=2,
                        label="Max New Tokens",
                        info="控制生成长度"
                    )
                    repetition_penalty_slider = gr.Slider(
                        minimum=1.01, maximum=1.5, value=DEFAULT_REPETITION_PENALTY, step=0.01,
                        label="Repetition Penalty",
                        info="控制重复程度"
                    )
                    
                    # 操作按钮
                    gr.Markdown("#### 操作")
                    clear_btn = gr.Button("清空历史", variant="secondary")
                    
                    # 使用说明
                    gr.Markdown("#### 使用说明")
                    gr.HTML("""
                    <div style="font-size: 12px; color: #666;">
                    <p><strong>参数建议：</strong></p>
                    <ul>
                    <li>Temperature: 0.7-1.0（平衡创造性与准确性）</li>
                    <li>Top-p: 0.8-0.95（控制词汇多样性）</li>
                    <li>Top-k: 40-60（限制候选词汇）</li>
                    <li>Max New Tokens: 512-2048（控制回答长度）</li>
                    <li>Repetition Penalty: 1.05-1.15（减少重复）</li>
                    </ul>
                    </div>
                    """)
        
        # 图像生成标签页
        with gr.TabItem("图像生成"):
            # 创建子标签页
            with gr.Tabs() as image_tabs:
                # 文生图子标签页
                with gr.TabItem("文生图"):
                    with gr.Row():
                        # 左侧：图像生成区域
                        with gr.Column(scale=2):
                            image_prompt = gr.Textbox(
                                label="图像描述",
                                placeholder="描述您想要生成的图像...",
                                lines=3
                            )
                            negative_prompt = gr.Textbox(
                                label="负面提示词（可选）",
                                placeholder="描述您不希望在图像中出现的内容...",
                                lines=2
                            )
                            generate_btn = gr.Button("生成图像", variant="primary")
                            generated_image = gr.Image(
                                label="生成的图像",
                                type="pil"
                            )
                        
                        # 右侧：参数设置
                        with gr.Column(scale=1):
                            gr.Markdown("### 文生图参数")
                            image_width = gr.Slider(
                                minimum=256, maximum=1024, value=512, step=64,
                                label="图像宽度"
                            )
                            image_height = gr.Slider(
                                minimum=256, maximum=1024, value=512, step=64,
                                label="图像高度"
                            )
                            num_inference_steps = gr.Slider(
                                minimum=10, maximum=50, value=DEFAULT_NUM_INFERENCE_STEPS, step=1,
                                label="推理步数"
                            )
                            guidance_scale = gr.Slider(
                                minimum=1.0, maximum=20.0, value=DEFAULT_GUIDANCE_SCALE, step=0.5,
                                label="引导强度"
                            )
                            save_image_btn = gr.Button("保存图像")
                
                # 图生图子标签页
                with gr.TabItem("图生图"):
                    with gr.Row():
                        # 左侧：图生图区域
                        with gr.Column(scale=2):
                            img2img_prompt = gr.Textbox(
                                label="图像描述",
                                placeholder="描述您想要生成的图像...",
                                lines=3
                            )
                            img2img_negative_prompt = gr.Textbox(
                                label="负面提示词（可选）",
                                placeholder="描述您不希望在图像中出现的内容...",
                                lines=2
                            )
                            init_image_upload = gr.Image(
                                label="上传参考图像",
                                type="pil"
                            )
                            generate_img2img_btn = gr.Button("生成图像", variant="primary")
                            img2img_generated_image = gr.Image(
                                label="生成的图像",
                                type="pil"
                            )
                        
                        # 右侧：图生图参数设置
                        with gr.Column(scale=1):
                            gr.Markdown("### 图生图参数")
                            
                            # 分辨率设置
                            gr.Markdown("#### 分辨率设置")
                            img2img_width = gr.Slider(
                                minimum=256, maximum=1024, value=512, step=64,
                                label="图像宽度"
                            )
                            img2img_height = gr.Slider(
                                minimum=256, maximum=1024, value=512, step=64,
                                label="图像高度"
                            )
                            
                            # 生成参数
                            gr.Markdown("#### 生成参数")
                            img2img_num_inference_steps = gr.Slider(
                                minimum=10, maximum=100, value=50, step=1,
                                label="推理步数",
                                info="步数越多质量越好，但耗时更长"
                            )
                            img2img_guidance_scale = gr.Slider(
                                minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                                label="引导强度",
                                info="控制提示词的影响程度"
                            )
                            img2img_strength = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.6, step=0.05,
                                label="变化强度",
                                info="值越高变化越大，值越低保留原图越多"
                            )
                            
                            # 使用说明
                            gr.Markdown("#### 使用说明")
                            gr.HTML("""
                            <div style="font-size: 12px; color: #666;">
                            <p><strong>提示词技巧：</strong></p>
                            <ul>
                            <li>使用具体的描述词汇</li>
                            <li>添加艺术风格关键词</li>
                            <li>使用负面提示词排除不想要的元素</li>
                            </ul>
                            <p><strong>参数建议：</strong></p>
                            <ul>
                            <li>推理步数：20-50（平衡质量与速度）</li>
                            <li>引导强度：7-12（控制提示词影响）</li>
                            <li>变化强度：0.3-0.8（控制变化程度）</li>
                            </ul>
                            </div>
                            """)
                            
                            save_img2img_btn = gr.Button("保存图像")
                
                # 图生文子标签页
                with gr.TabItem("图生文"):
                    with gr.Row():
                        # 左侧：图生文区域
                        with gr.Column(scale=2):
                            caption_image_upload = gr.Image(
                                label="上传图像",
                                type="pil"
                            )
                            generate_caption_btn = gr.Button("生成描述", variant="primary")
                            generated_caption = gr.Textbox(
                                label="生成的图像描述",
                                placeholder="图像描述将在这里显示...",
                                lines=4,
                                interactive=True
                            )
                            
                            # 描述改写区域
                            gr.Markdown("### 描述改写")
                            rewrite_prompt = gr.Textbox(
                                label="改写要求",
                                placeholder="例如：用更诗意的语言描述，或者用专业术语描述...",
                                lines=2
                            )
                            rewrite_caption_btn = gr.Button("改写描述", variant="secondary")
                            rewritten_caption = gr.Textbox(
                                label="改写后的描述",
                                placeholder="改写后的描述将在这里显示...",
                                lines=4,
                                interactive=True
                            )
                        
                        # 右侧：图生文参数设置
                        with gr.Column(scale=1):
                            gr.Markdown("### 图生文参数")
                            max_caption_length = gr.Slider(
                                minimum=20, maximum=200, value=DEFAULT_MAX_CAPTION_LENGTH, step=10,
                                label="最大描述长度"
                            )
                            gr.Markdown("### 使用说明")
                            gr.HTML("""
                            <div style="font-size: 12px; color: #666;">
                            <p><strong>生成描述：</strong>上传图像后点击"生成描述"按钮</p>
                            <p><strong>改写描述：</strong>在"改写要求"中输入具体要求，然后点击"改写描述"</p>
                            <p><strong>中文转换：</strong>在改写要求中输入"翻译成中文"或"用中文描述"</p>
                            <p><strong>改写示例：</strong></p>
                            <ul>
                            <li>翻译成中文</li>
                            <li>用更诗意的语言描述</li>
                            <li>用专业摄影术语描述</li>
                            <li>用简洁的语言概括</li>
                            <li>添加情感色彩</li>
                            </ul>
                            </div>
                            """)

    # 状态管理
    state = gr.State([])
    file_context_state = gr.State("")

    # 绑定事件
    # 文件上传处理
    file_upload.change(
        handle_file_upload,
        inputs=[file_upload],
        outputs=[file_preview, file_context_state]
    )
    
    # 清除文件
    file_clear_btn.click(
        lambda: ("", ""),
        inputs=None,
        outputs=[file_preview, file_context_state]
    )
    
    # 回车chat_interface
    user_input.submit(
        chat_interface, 
        inputs=[user_input, state, system_message, top_p_slider, top_k_slider, temperature_slider, max_new_tokens_slider, repetition_penalty_slider, file_context_state], 
        outputs=[chatbot, state, system_message, user_input, file_context_state],
        queue=True
    )
    # 发送chat_interface
    send_btn.click(
        chat_interface, 
        inputs=[user_input, state, system_message, top_p_slider, top_k_slider, temperature_slider, max_new_tokens_slider, repetition_penalty_slider, file_context_state], 
        outputs=[chatbot, state, system_message, user_input, file_context_state],
        queue=True
    )
    clear_btn.click(
        clear_history, 
        inputs=None, 
        outputs=[chatbot, state, system_message, user_input, file_context_state, repetition_penalty_slider],
        queue=True
    )
    
    # 图像生成事件绑定
    generate_btn.click(
        handle_image_generation,
        inputs=[image_prompt, negative_prompt, image_width, image_height, num_inference_steps, guidance_scale],
        outputs=[generated_image, gr.Textbox(label="状态", visible=False)],
        queue=True
    )
    
    save_image_btn.click(
        save_generated_image,
        inputs=[generated_image],
        outputs=[gr.Textbox(label="保存状态", visible=False)],
        queue=True
    )
    
    # 图生图事件绑定
    generate_img2img_btn.click(
        handle_img2img_generation,
        inputs=[img2img_prompt, init_image_upload, img2img_negative_prompt, img2img_num_inference_steps, img2img_guidance_scale, img2img_strength, img2img_width, img2img_height],
        outputs=[img2img_generated_image, gr.Textbox(label="状态", visible=False)],
        queue=True
    )
    
    save_img2img_btn.click(
        save_generated_image,
        inputs=[img2img_generated_image],
        outputs=[gr.Textbox(label="保存状态", visible=False)],
        queue=True
    )
    
    # 图生文事件绑定
    generate_caption_btn.click(
        handle_image_caption_generation,
        inputs=[caption_image_upload, max_caption_length],
        outputs=[generated_caption, gr.Textbox(label="状态", visible=False)],
        queue=True
    )
    
    rewrite_caption_btn.click(
        handle_caption_rewrite,
        inputs=[generated_caption, rewrite_prompt],
        outputs=[rewritten_caption, gr.Textbox(label="状态", visible=False)],
        queue=True
    )

    # JS
    gr.HTML("""
    <script>
        function scrollChat() {
            const chatContainer = document.getElementById('chat-container');
            if(chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }

        const observer = new MutationObserver(scrollChat);
        const chatContainer = document.getElementById('chat-container');
        if(chatContainer) {
            observer.observe(chatContainer, { childList: true, subtree: true });
        }
    </script>
    """)

# 初始化模型属性
generate_image.pipe = None
generate_img2img.pipe = None
generate_image_caption.processor = None
generate_image_caption.model = None

print("\n" + "="*50)
print("正在启动 WebUI 服务...")
print("="*50 + "\n")
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
