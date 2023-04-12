import pprint
from typing import List

import pyrallis
import torch
from PIL import Image

from config import BatchConfig
from pipeline_attend_and_excite import AttendAndExcitePipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
import json
import time

def load_model():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    stable = AttendAndExcitePipeline.from_pretrained("./models/stable-diffusion-v1-4").to(device)
    return stable

def run_on_prompt(prompt: List[str],
                  model: AttendAndExcitePipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  condition_images_path : str,
                  seed: torch.Generator,
                  config: BatchConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    condition_images_path = condition_images_path,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd= config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    # negative_prompt="Cartoon, drawing, bad quality, fuzzy, blurry",
                    # negative_prompt="not cat, not frog"
                    negative_prompt="bad result, worst, random, invalid, inaccurate, imperfect, blurry, deformed, disfigured, mutation, mutated, ugly, out of focus, bad anatomy, text, error, extra digit, fewer digits, worst quality, low quality, normal quality, noise, jpeg artifact, compression artifact, signature, watermark, username, logo, low resolution, worst resolution, bad resolution, normal resolution, bad detail, bad details, bad lighting, bad shadow, bad shading, bad background, worst background."
                    )
    image = outputs.images[0]
    return image

@pyrallis.wrap()
def main(config: BatchConfig):
    stable = load_model()
    with open(config.prompts, 'r') as f:
        prompts = json.load(f)['prompts']

    for entry in prompts:
        prompt = entry["prompt"]
        token_indices = entry["token_indices"]
        condition_images_path = None
        if "condition_images_path" in entry.keys():
            condition_images_path = entry["condition_images_path"]
        for seed in config.seeds:
            print(f"Seed: {seed}")
            import random
            rand_seed = random.randint(0,99999999)
            g = torch.Generator('cuda').manual_seed(rand_seed)
            controller = AttentionStore()
            image = run_on_prompt(prompt=prompt,
                                  model=stable,
                                  controller=controller,
                                  token_indices=token_indices,
                                  condition_images_path=condition_images_path,
                                  seed=g,
                                  config=config)
            prompt_output_path = config.output_path / prompt[:20]
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            image.save(prompt_output_path / f'{seed}_{time.time()}.png')


if __name__ == '__main__':
    main()