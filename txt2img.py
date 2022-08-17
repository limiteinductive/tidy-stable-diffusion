import argparse
import os
from itertools import islice

import numpy as np
import torch
from einops import rearrange
from tsd.autoencoder import AutoencoderKL, FrozenCLIPEmbedder
from tsd.sampling import LatentDiffusion, PLMSSampler
from PIL import Image
from tsd.utils import seed_everything
from tqdm import tqdm, trange


def load_model_from_checkpoint(model, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="on which device to execute the model on",
    )

    opt = parser.parse_args()

    C = 4
    F = 8

    seed_everything(opt.seed)

    print("loading autoencoder")
    autoencoder = torch.load("sd_autoencoder.pt")  # AutoencoderKL(ddconfig=ddconfig, embed_dim=4)

    # load_model_from_checkpoint(autoencoder, "sd_autoencoder.pt")
    print("loading embedder")
    # embedder = FrozenCLIPEmbedder().eval()
    # load_model_from_checkpoint(embedder, "sd_embedder.pt")
    embedder = torch.load("sd_embedder.pt")

    print("loading model")
    model = LatentDiffusion(
        autoencoder,
        embedder,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps_cond=1,
        log_every_t=200,
        timesteps=1000,
        first_stage_key="jpg",
        cond_stage_key="txt",
        image_size=64,
        channels=4,
        cond_stage_trainable=False,
        scale_factor=0.18215,
        device=opt.device,
    )
    

    unet = torch.load("sd_unet.pt")
    #  model.model.diffusion_model
    # load_model_from_checkpoint(unet, "./sd_unet.pt")

    model = model.to(opt.device)

    sampler = PLMSSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    assert opt.prompt is not None
    data = [opt.batch_size * [opt.prompt]]

    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            for _ in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(opt.batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [C, opt.H // F, opt.W // F]
                    samples_ddim, _ = sampler.sample(
                        S=opt.ddim_steps,
                        conditioning=c,
                        batch_size=opt.batch_size,
                        shape=shape,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=None,
                    )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp(
                        (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                    )

                    for x_sample in x_samples_ddim:
                        sample_path = "outputs/txt2img-samples/samples"
                        base_count = len(os.listdir(sample_path))
                        x_sample = 255.0 * rearrange(
                            x_sample.cpu().numpy(), "c h w -> h w c"
                        )
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{base_count:05}.png")
                        )
                        base_count += 1



if __name__ == "__main__":
    main()
