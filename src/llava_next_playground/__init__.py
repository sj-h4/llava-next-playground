import argparse

from transformers import TextStreamer
import cv2
from PIL import Image
import torch
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200


def sample_frames(video_file_name: str, num_frames: int):
    """
    Sample a video

    Copied from https://github.com/LLaVA-VL/LLaVA-NeXT/blob/inference/playground/demo/interleave_demo.py
    Args:
        video_file_name: a file path of the video
        num_frames: number of farames

    Returns:
        A list of `Image` of the input video
    """
    video = cv2.VideoCapture(video_file_name)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", default="data/sample.mp4", type=str)
    parser.add_argument("--num-frames", default=15, type=int)
    args = parser.parse_args()

    video_filename = args.file_path
    num_frames = (
        args.num_frames
    )  # 実際に入力するフレーム数。このフレーム数に動画をサンプリングする

    model_id = "lmms-lab/llava-next-interleave-7b"
    model_base = None
    model_name = "llava_qwen"
    image_processor: SigLipImageProcessor  # このモデルはこっち （Clip のほうじゃない） ref: https://huggingface.co/lmms-lab/llava-next-interleave-7b/blob/dfa28e80fd0881edd488cd36582878cd2f9c2fb4/config.json
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_id, model_base=model_base, model_name=model_name
    )

    frames = sample_frames(video_filename, num_frames)
    image_list = frames

    image_tensor = [
        image_processor.preprocess(f, return_tensors="pt")["pixel_values"][0]
        .half()
        .to(model.device)
        for f in image_list
    ]
    image_tensor = torch.stack(image_tensor)
    image_token = DEFAULT_IMAGE_TOKEN * num_frames

    prompt_text = input("Enter your prompt:\n")
    if prompt_text is None:
        prompt_text = "Describe this video."
    input_text = image_token + prompt_text

    conv_mode = "qwen_1_5"
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], input_text)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    stop_str = (
        conversation.sep
        if conversation.sep_style != SeparatorStyle.TWO
        else conversation.sep2
    )
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            streamer=streamer,
            use_cache=False,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.decode(output_ids[0]).strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]

    print("Video: ", video_filename)
    print("prompt: ", prompt_text)
    print("outputs: ", outputs)
    print("=====\n")
    return 0
