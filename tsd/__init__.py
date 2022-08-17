import torch
from pathlib import Path


def load_models(path: str=".", device="cuda"):
    path = Path(path)
    autoencoder = torch.load(path / "sd_autoencoder.pt", map_location=device)
    embedder = torch.load(path / "sd_embedder.pt", map_location=device)
    model = torch.load(path / "sd_model.pt", map_location=device)

    return {"autoencoder": autoencoder, "embedder": embedder, "model": model}

