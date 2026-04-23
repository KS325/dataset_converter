
import torch
import imageio
import numpy as np
from glob import glob
import json
from box import Box
import yaml
import os
from tqdm import tqdm

def load_config():
    # Config.yaml の呼び出し
    with open("./conf/config_n.yaml", "r") as yml:
        cfg = Box(yaml.safe_load(yml))
    return cfg


def save_gif(obs, cam_name, save_path, fps=10):
    imgs = obs["camera"][cam_name]  # (T, C, H, W)
    # print(imgs.max(), imgs.min())

    frames = []
    for i in range(len(imgs)):
        img = imgs[i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)

        img = (img * 255.0).astype(np.uint8)
        frames.append(img)

    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    # print(f"Saved GIF to {save_path}")

def main():
    cam_list = None
    top_root = "/home/sato/data/pt_datasets/KS325/"
    epi_list = sorted(glob(f"{top_root}/episode--*"))
    # print(epi_list[0])
    for epi in tqdm(epi_list):
        obs = torch.load(f"{epi}/obs.pt", map_location="cpu")

        if cam_list is None:
            cam_list = obs["camera"].keys()

        for cam_name in cam_list:
            save_path = os.path.join(epi, f"{cam_name}.gif")
            save_gif(obs, cam_name, save_path, 10)


if __name__ == "__main__":
    main()


