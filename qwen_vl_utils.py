from typing import List, Tuple
from PIL import Image


def process_vision_info(messages: List[dict]) -> Tuple[list, list | None]:
    """Collect image inputs from the messages for the processor."""
    image_inputs = []
    video_inputs = None
    for msg in messages:
        for part in msg.get("content", []):
            if part.get("type") == "image":
                path = part.get("image")
                if isinstance(path, str):
                    image_inputs.append(Image.open(path))
                elif isinstance(path, Image.Image):
                    image_inputs.append(path)
    return image_inputs, video_inputs