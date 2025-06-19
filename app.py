#!/usr/bin/env python

import os
import re
import tempfile
from threading import Thread
from collections.abc import Iterator

import cv2
import gradio as gr
import spaces
import torch
from loguru import logger
from PIL import Image
from transformers import (
    AutoProcessor,
    TextIteratorStreamer,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

# Constants
MODEL_NAME = "lingshu-medical-mllm/Lingshu-7B"
MAX_NUM_IMAGES = int(os.getenv("MAX_NUM_IMAGES", "5"))

# System prompt optimized for GP Assistant role
SYSTEM_PROMPT = (
    "You are a professional General Practitioner (GP) Assistant powered by MedLlama-3-8B, designed to provide clinically relevant, evidence-based, and ethical medical assistance. "
    "Your responses should align with modern medical guidelines (e.g., WHO, CDC, NICE, AAFP) and best practices for primary care.\n\n"
    "Your role is to:\n"
    "- Assist GPs, healthcare professionals, and medical staff in diagnosing, managing, and treating patients.\n"
    "- Provide concise, structured, and medically accurate responses.\n"
    "- Support clinical decision-making while emphasizing patient safety and ethical considerations.\n"
    "- Use formal, professional language, but ensure clarity and accessibility for different audiences.\n\n"
    "Response Structure Guidelines:\n"
    "- Summary of Inquiry: Briefly rephrase the user’s question for context.\n"
    "- Clinical Insights & Evidence: Provide the latest clinical knowledge relevant to the case.\n"
    "- Next Steps & Recommendations: Offer guidance for further evaluation, treatment options, and clinical pathways.\n"
    "- Red Flags & Referral Advice: Highlight urgent signs that require immediate medical attention.\n"
    "- References (if applicable): Cite trusted medical sources when giving specific recommendations."
)

# Model and processor initialization
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


# Utility functions for media handling and history conversion

def count_files(paths: list[str]) -> tuple[int, int]:
    images = sum(1 for p in paths if not p.endswith(".mp4"))
    videos = sum(1 for p in paths if p.endswith(".mp4"))
    return images, videos


def validate_media_constraints(message: dict, history: list[dict]) -> bool:
    new_imgs, new_vids = count_files(message.get("files", []))
    hist_imgs, hist_vids = count_files([item["content"] for item in history if item.get("files")])
    total_imgs, total_vids = hist_imgs + new_imgs, hist_vids + new_vids

    if total_vids > 1:
        gr.Warning("Only one video is supported.")
        return False
    if total_vids == 1 and total_imgs > 0:
        gr.Warning("Mixing images and videos is not allowed.")
        return False
    if total_vids == 0 and total_imgs > MAX_NUM_IMAGES:
        gr.Warning(f"You can upload up to {MAX_NUM_IMAGES} images.")
        return False
    if message.get("text", "").count("<image>") != new_imgs:
        gr.Warning("<image> tags must match the number of uploaded images.")
        return False
    return True


def downsample_video(video_path: str) -> list[tuple[Image.Image, float]]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(total_frames // MAX_NUM_IMAGES, 1)

    frames = []
    for i in range(0, min(total_frames, MAX_NUM_IMAGES * interval), interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        frames.append((pil_img, round(i / fps, 2)))
    cap.release()
    return frames


def process_video(video_path: str) -> list[dict]:
    content = []
    for img, ts in downsample_video(video_path):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            content.extend([
                {"type": "text", "text": f"Frame {ts}s:"},
                {"type": "image", "image": tmp.name},
            ])
    logger.debug(f"Processed video content: {content}")
    return content


def process_images_in_text(message: dict) -> list[dict]:
    parts = re.split(r"(<image>)", message["text"])
    content, idx = [], 0
    for part in parts:
        if part == "<image>":
            content.append({"type": "image", "image": message["files"][idx]})
            idx += 1
        elif part.strip():
            content.append({"type": "text", "text": part.strip()})
    return content


def assemble_message(message: dict, history: list[dict]) -> list[dict]:
    if not message.get("files"):
        return [{"type": "text", "text": message.get("text", "")}]
    if message["files"][0].endswith(".mp4"):
        return [{"type": "text", "text": message["text"]}] + process_video(message["files"][0])
    if "<image>" in message.get("text", ""):
        return process_images_in_text(message)
    return ([{"type": "text", "text": message["text"]}] +
            [{"type": "image", "image": p} for p in message["files"]])


def convert_history(history: list[dict]) -> list[dict]:
    convo = []
    buffer = []
    for msg in history:
        role, content = msg["role"], msg.get("content")
        if role == "assistant":
            if buffer:
                convo.append({"role": "user", "content": buffer})
                buffer = []
            convo.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
        else:
            if isinstance(content, str):
                buffer.append({"type": "text", "text": content})
            else:
                buffer.append({"type": "image", "image": content[0]})
    return convo


@spaces.GPU(duration=120)
def run(
    message: dict,
    history: list[dict],
    system_prompt: str = SYSTEM_PROMPT,
    max_new_tokens: int = 2048,
) -> Iterator[str]:
    if not validate_media_constraints(message, history):
        yield ""
        return

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.extend(convert_history(history))
    messages.append({"role": "user", "content": assemble_message(message, history)})

    # Prepare inputs for model
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        temperature=0.7,
        top_p=1,
        repetition_penalty=1,
    )
    Thread(target=model.generate, kwargs=gen_kwargs).start()

    output = ""
    for delta in streamer:
        output += delta
        yield output

# Demo interface
DESCRIPTION = """
This is a demo of a professional General Practitioner (GP) Assistant. It delivers clinically relevant, evidence-based, and ethical medical guidance. \n\n"""
DESCRIPTION += (
    "Capabilities: Multimodal understanding of medical images across modalities, including X-Ray, CT, MRI, Ultrasound, Histopathology, Dermoscopy, and more.\n"
    "Designed for primary care workflows, supporting diagnosis, treatment planning, and patient education."
)


demo = gr.ChatInterface(
    fn=run,
    type="messages",
    chatbot=gr.Chatbot(type="messages", scale=1, allow_tags=["image"]),
    textbox=gr.MultimodalTextbox(
        file_types=["image", ".mp4"],
        file_count="multiple",
        autofocus=True,
    ),
    multimodal=True,
    additional_inputs=[
        gr.Textbox(label="System Prompt", value=SYSTEM_PROMPT, lines=8),
        gr.Slider(
            label="Max New Tokens", minimum=100, maximum=8192, step=10, value=2048
        ),
    ],
    stop_btn=False,
    title="GP Assistant",
    description=DESCRIPTION,
    run_examples_on_click=False,
    cache_examples=False,
    css_paths="style.css",
    delete_cache=(1800, 1800),
)

if __name__ == "__main__":
    demo.launch(share=True)
