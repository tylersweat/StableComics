import requests
import json
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

# load the pipeline
device = "cuda"
model_id_or_path = "../models/stable-diffusion-v1-4"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)

# or download via git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
# and pass `model_id_or_path="./stable-diffusion-v1-5"`.
pipe = pipe.to(device)


with open('batch_img2img.json', 'r') as f:
    batch = json.load(f)['batch']

for entry in batch:
    init_image = Image.open(entry['input']).convert('RGB').resize((512, 512))

    images = pipe(prompt=entry['prompt'],
                  image=init_image,
                  strength=entry['strength'],
                  guidance_scale=entry['guidance_scale']).images

    images[0].save(entry['output'])

# # let's download an initial image
# url = "./inputs/1.jpg"

# # response = requests.get(url)
# # init_image = Image.open(BytesIO(response.content)).convert("RGB")
# init_image = Image.open(url).convert("RGB")
# init_image = init_image.resize((768, 512))

# prompt = "A fantasy landscape, trending on artstation"

# images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

# images[0].save("./outputs/fantasy_landscape.png")