from typing import List

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from tsd.utils import seed_everything

from .sampling import PLMSSampler


def run_stable_diffusion(
    model,
    prompt: str,
    steps=50,
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    guidance_scale: int = 7.5,
    seed: int = None,
) -> List[np.ndarray]:

    seed_everything(seed)
    sampler = PLMSSampler(model)

    data = [batch_size * [prompt]]
    outputs= []

    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            for prompts in tqdm(data, desc="data"):
                uc = None
                if guidance_scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)
                shape = [4, height // 8, width // 8]
                samples_ddim, _ = sampler.sample(
                    S=steps,
                    conditioning=c,
                    batch_size=batch_size,
                    shape=shape,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=uc,
                    eta=0,
                    x_T=None,
                )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )

                for x_sample in x_samples_ddim:
                    array = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    image = Image.fromarray(array.astype(np.uint8))
                    outputs.append(image)
            
    return outputs


def save_samples(samples: List[np.ndarray], path: str) -> None:
    for i, sample in enumerate(samples):
        sample.save(f"{path}/{i}.png")



