
import matplotlib.pyplot as plt
import torch
from glob import glob
from tqdm import tqdm
import json
from box import Box
import yaml
import numpy as np
import os

def load_config():
    # Config.yaml の呼び出し
    with open("./conf/config_n.yaml", "r") as yml:
        cfg = Box(yaml.safe_load(yml))
    return cfg

def load_datas(path_list):
    all_data = []
    for path in path_list:
        joint_data = torch.load(f"{path}/obs.pt")["action"]
        all_data.append(joint_data.cpu().numpy())

    return all_data

def joint_plot(predictions: list, targets:list, save_path: str | None=None , all_data: bool | None=False):
    prediction_datas = load_datas(predictions)
    target_data = load_datas(targets)[0]

    for epi, data in enumerate(tqdm(prediction_datas)):
        plt.clf()
        fig, ax = plt.subplots(data.shape[-1], 1, figsize=(12, 8))

        for i in range(data.shape[-1]):
            ax[i].plot(data[:, i], color="C0", label="prediction")
            ax[i].plot(target_data[:, i], color="C1", label="target")
            ax[i].set_ylim([-1.1, 1.1])
            ax[i].legend(loc="upper right")

        if not all_data:
            fig.savefig(f"{predictions[epi]}/joint_result.png", dpi=300, bbox_inches="tight")

    if all_data is True and save_path is not None:
        fig.savefig(f"{save_path}/joint_result.png", dpi=300, bbox_inches="tight")
    pass

def plot_all_episodes(root: str):
    paths = sorted(glob(f"{root}/episode--*"))
    datas = load_datas(paths)

    fig, ax = plt.subplots(datas[0].shape[-1], 1, figsize=(12, 8))

    for epi, data in enumerate(tqdm(datas)):

        for i in range(data.shape[-1]):
            ax[i].plot(data[:, i], color="C0")
            ax[i].set_ylim([-1.1, 1.1])
    fig.savefig(f"{root}/.meta/joint_plot.png", dpi=300, bbox_inches="tight")


def main():
    root = "/home/sato/data/pt_datasets/KS325/close-lower-drawer"
    prediction_paths = sorted(glob(f"{root}/episode--*"))
    target_paths = sorted(glob("/home/sato/data/pt_datasets/KS325/skill-set/episode--165"))
    save_path = root

    # joint_plot(prediction_paths, target_paths, save_path)
    plot_all_episodes(root)
    pass

if __name__ == "__main__":
    main()

