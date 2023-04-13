import json
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import logging
import clip
import numpy as np
from lavis.models import load_model_and_preprocess
import os

num_generations = 10

negative_prompt = "bad, blurry, deformed, disfigured, cartoon, text, logo, ugly, terrible, awful, grainy"
cartoonify_positive_prompt = "a high quality anime drawing"
cartoonify_negative_prompt = "bad result, worst, random, invalid, inaccurate, imperfect, blurry, deformed, disfigured, mutation, mutated, ugly, out of focus, bad anatomy, text, error, extra digit, fewer digits, worst quality, low quality, normal quality, noise, jpeg artifact, compression artifact, signature, watermark, username, logo, low resolution, worst resolution, bad resolution, normal resolution, bad detail, bad details, bad lighting, bad shadow, bad shading, bad background, worst background."

outputs_dir = './outputs'

logging.disable_progress_bar()

img2img_strength = 0.95
img2img_guidance_scale = 7.5

model = "../models/sd_v2-1_euler_a"
device = "cuda"

# Define SD scheduler.
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model, subfolder='scheduler')

# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, scheduler=euler_a, safety_checker=None, requires_safety_checker=False)
pipe.to(device)

cartoonify_model = '../models/Cartoonify'
cartoonify = StableDiffusionImg2ImgPipeline.from_pretrained(cartoonify_model, torch_dtype=torch.float16, safety_checker=None)
cartoonify.to(device)

with open('comics.json', 'r') as f:
    comics = json.load(f)['comics']

for comic in comics:
    for frame, n in zip(comic['frames'], range(3)):
        frame_outputs_dir = f'{outputs_dir}/{comic["id"]}/{n}'
        if not os.path.exists(frame_outputs_dir):
            os.makedirs(frame_outputs_dir)
        # Generate `num_generations` versions of each frame, for a human to pick the best later.
        result = pipe(prompt=frame['prompt'], negative_prompt=negative_prompt, num_images_per_prompt=num_generations)
        images = result.images
        print(result.nsfw_content_detected)
        # Save the generations.
        for i in range(num_generations):
            image = images[i]
            image_cartoon = cartoonify(prompt=cartoonify_positive_prompt, image=image, strength=0.5, guidance_scale=7.5, negative_prompt=cartoonify_negative_prompt).images[0]
            # Save the original and cartoonified outputs.
            image.save(f'{frame_outputs_dir}/{i:02d}_original.png')
            image_cartoon.save(f'{frame_outputs_dir}/{i:02d}_cartoon.png')

print('Generations complete! :)')