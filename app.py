#!/usr/bin/env python

import os
import re
import tempfile
from collections.abc import Iterator
from threading import Thread

import cv2
import gradio as gr
import spaces
import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

#import subprocess
#subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "lingshu-medical-mllm/Lingshu-7B",
    torch_dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("lingshu-medical-mllm/Lingshu-7B")


MAX_NUM_IMAGES = int(os.getenv("MAX_NUM_IMAGES", "5"))


MAX_NUM_IMAGES = int(os.getenv("MAX_NUM_IMAGES", "5"))


def count_files_in_new_message(paths: list[str]) -> tuple[int, int]:
    image_count = 0
    video_count = 0
    for path in paths:
        if path.endswith(".mp4"):
            video_count += 1
        else:
            image_count += 1
    return image_count, video_count


def count_files_in_history(history: list[dict]) -> tuple[int, int]:
    image_count = 0
    video_count = 0
    for item in history:
        if item["role"] != "user" or isinstance(item["content"], str):
            continue
        if item["content"][0].endswith(".mp4"):
            video_count += 1
        else:
            image_count += 1
    return image_count, video_count


def validate_media_constraints(message: dict, history: list[dict]) -> bool:
    new_image_count, new_video_count = count_files_in_new_message(message["files"])
    history_image_count, history_video_count = count_files_in_history(history)
    image_count = history_image_count + new_image_count
    video_count = history_video_count + new_video_count
    if video_count > 1:
        gr.Warning("Only one video is supported.")
        return False
    if video_count == 1:
        if image_count > 0:
            gr.Warning("Mixing images and videos is not allowed.")
            return False
        if "<image>" in message["text"]:
            gr.Warning("Using <image> tags with video files is not supported.")
            return False
    if video_count == 0 and image_count > MAX_NUM_IMAGES:
        gr.Warning(f"You can upload up to {MAX_NUM_IMAGES} images.")
        return False
    if "<image>" in message["text"] and message["text"].count("<image>") != new_image_count:
        gr.Warning("The number of <image> tags in the text does not match the number of images.")
        return False
    return True


def downsample_video(video_path: str) -> list[tuple[Image.Image, float]]:
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(total_frames // MAX_NUM_IMAGES, 1)
    frames: list[tuple[Image.Image, float]] = []

    for i in range(0, min(total_frames, MAX_NUM_IMAGES * frame_interval), frame_interval):
        if len(frames) >= MAX_NUM_IMAGES:
            break

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))

    vidcap.release()
    return frames


def process_video(video_path: str) -> list[dict]:
    content = []
    frames = downsample_video(video_path)
    for frame in frames:
        pil_image, timestamp = frame
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            pil_image.save(temp_file.name)
            content.append({"type": "text", "text": f"Frame {timestamp}:"})
            content.append({"type": "image", "image": temp_file.name})
    logger.debug(f"{content=}")
    return content


def process_interleaved_images(message: dict) -> list[dict]:
    logger.debug(f"{message['files']=}")
    parts = re.split(r"(<image>)", message["text"])
    logger.debug(f"{parts=}")

    content = []
    image_index = 0
    for part in parts:
        logger.debug(f"{part=}")
        if part == "<image>":
            content.append({"type": "image", "image": message["files"][image_index]})
            logger.debug(f"file: {message['files'][image_index]}")
            image_index += 1
        elif part.strip():
            content.append({"type": "text", "text": part.strip()})
        elif isinstance(part, str) and part != "<image>":
            content.append({"type": "text", "text": part})
    logger.debug(f"{content=}")
    return content


def process_new_user_message(message: dict) -> list[dict]:
    if not message["files"]:
        return [{"type": "text", "text": message["text"]}]
    if message["files"][0].endswith(".mp4"):
        return [{"type": "text", "text": message["text"]}, *process_video(message["files"][0])]
    if "<image>" in message["text"]:
        return process_interleaved_images(message)
    return [
        {"type": "text", "text": message["text"]},
        *[{"type": "image", "image": path} for path in message["files"]],
    ]


def process_history(history: list[dict]) -> list[dict]:
    messages = []
    current_user_content: list[dict] = []
    for item in history:
        if item["role"] == "assistant":
            if current_user_content:
                messages.append({"role": "user", "content": current_user_content})
                current_user_content = []
            messages.append({"role": "assistant", "content": [{"type": "text", "text": item["content"]}]})
        else:
            content = item["content"]
            if isinstance(content, str):
                current_user_content.append({"type": "text", "text": content})
            else:
                current_user_content.append({"type": "image", "image": content[0]})
    return messages


@spaces.GPU(duration=120)
def run(message: dict, history: list[dict], system_prompt: str = "", max_new_tokens: int = 2048) -> Iterator[str]:
    if not validate_media_constraints(message, history):
        yield ""
        return

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.extend(process_history(history))
    messages.append({"role": "user", "content": process_new_user_message(message)})

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    streamer = TextIteratorStreamer(processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        temperature=0.7,
        top_p=1,
        repetition_penalty=1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    output = ""
    for delta in streamer:
        output += delta
        yield output


DESCRIPTION = """\
This is a demo of Lingshu 7B, a multimodal model trained for performance on medical text and image comprehension.
Lingshu supports more than 12 medical imaging modalities, including X-Ray, CT Scan, MRI, Microscopy, Ultrasound, Histopathology, Dermoscopy, Fundus, OCT, Digital Photography, Endoscopy, and PET.
"""

demo = gr.ChatInterface(
    fn=run,
    type="messages",
    chatbot=gr.Chatbot(type="messages", scale=1, allow_tags=["image"]),
    textbox=gr.MultimodalTextbox(file_types=["image", ".mp4"], file_count="multiple", autofocus=True),
    multimodal=True,
    additional_inputs=[
        gr.Textbox(label="System Prompt", value="You are a helpful medical expert."),
        gr.Slider(label="Max New Tokens", minimum=100, maximum=8192, step=10, value=2048),
    ],
    stop_btn=False,
    title="Lingshu 7B",
    description=DESCRIPTION,
    run_examples_on_click=False,
    cache_examples=False,
    css_paths="style.css",
    delete_cache=(1800, 1800),
)

if __name__ == "__main__":
    demo.launch()
