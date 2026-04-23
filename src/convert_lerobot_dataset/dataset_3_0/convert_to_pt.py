
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import torch
import json
from glob import glob
import imageio
import matplotlib.pyplot as plt

class LerobotDataset2Pt():
    def __init__(self, config):
        self.cfg = config
        self.output_root, self.output_root_meta = self._generate_output_root()
        self.dataset, self.aciton_min, self.action_max = self._load_dataset(self.output_root_meta)
        pass

    def _generate_output_root(self, ):
        output_root = Path(os.path.join(self.cfg.dataset.output.root, self.cfg.dataset.input.data_name))
        output_root_meta = Path(os.path.join(output_root, ".meta"))
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(output_root_meta, exist_ok=True)

        return output_root, output_root_meta

    def _extract_action_range(self, dataset, root_meta):
        if self.cfg.preprocess.action.range is not None:
            print(f"Use '{self.cfg.preprocess.action.range.split("/")[:-1]}' action range.")
            range_path = Path(os.path.join(self.cfg.dataset.output.root, self.cfg.preprocess.action.range))
            action_min = torch.load(f"{range_path}/action_min.pt")
            action_max = torch.load(f"{range_path}/action_max.pt")
            print(f"action min: {action_min}\naction max: {action_max}")
        else:
            print("Extract 'action (min, max)'.")
            # actions = np.stack([
            #     dataset[i]["action"] for i in range(len(dataset))
            #     ]
            # )
            action_min = None
            action_max = None

            for i in tqdm(range(len(dataset))):
                action = dataset[i]["action"]
                if action_min is None:
                    action_min = action.clone()
                    action_max = action.clone()
                else:
                    action_min = torch.minimum(action_min, action)
                    action_max = torch.maximum(action_max, action)

            # action_min = torch.from_numpy(action_min).float()
            # action_max = torch.from_numpy(action_max).float()
            action_min = action_min.float()
            action_max = action_max.float()

            torch.save(action_min, f"{root_meta}/action_min.pt")
            torch.save(action_max, f"{root_meta}/action_max.pt")

            print(f"action min: {action_min}\naction max: {action_max}")

        return action_min, action_max

    def _load_dataset(self, root_meta):
        root = Path(os.path.join(self.cfg.dataset.input.root, self.cfg.dataset.input.data_name))
        print(f"Loading the lerobot dataset from {root}...")

        dataset = LeRobotDataset(
            repo_id="", 
            root = root
        )
        print(f"\t-> Load the dataset: '{self.cfg.dataset.input.data_name}.'")

        action_min, action_max = self._extract_action_range(dataset, root_meta)

        return dataset, action_min, action_max

    def generate_episode_group(self, dataset):
        print("Grouping by episode...")

        episode_map = {}
        for i in tqdm(range(len(dataset))):
            epi = int(dataset[i]["episode_index"])
            episode_map.setdefault(epi, []).append(i)

        episode_list = sorted(episode_map.keys())

        print(f"\t-> Grouped to {len(episode_list)} episodes.")

        return episode_map, episode_list

    def normalize(self, sample, min, max):
        return 2 * (sample - min) / (max - min + 1.0e-5) - 1

    def rename_camera(self, cam_name):
        return cam_name.replace("observation.", "")

    def _convert_episode(self, dataset, episode_map, episode_list):
        print(f"Start converting {len(episode_list)} episodes...")

        for epi in tqdm(episode_list):
            indices = episode_map[epi]

            actions = []
            images = {}
            task = None

            for idx in indices[::self.cfg.preprocess.downsample.step]:
                sample = dataset[idx]
                actions.append(sample["action"])

                for key, value in sample.items():
                    if key.startswith("observation.images"):
                        cam_name = key.split(".")[-1]
                        # print(value.shape)
                        img = value.float()

                        if self.cfg.preprocess.image.normalize:
                            img = img / 255.0

                        if self.cfg.preprocess.image.resize is not None:
                            img = torch.nn.functional.interpolate(
                                img.unsqueeze(0), 
                                size=self.cfg.preprocess.image.resize, 
                                mode="bilinear", 
                                align_corners=False, 
                            ).squeeze(0)

                        images.setdefault(cam_name, []).append(img.unsqueeze(0))

                if task is None:
                    task = sample["task"]

            actions = torch.from_numpy(np.stack(actions)).float()

            if self.cfg.preprocess.action.normalize:
                actions = self.normalize(actions, self.aciton_min, self.action_max)

            for cam_name in images:
                images[cam_name] = torch.cat(images[cam_name], dim=0)

            mask = torch.ones(
                len(indices[::self.cfg.preprocess.downsample.step]), 
                dtype=torch.bool
            )

            save_dir = f"{self.output_root}/episode--{epi:05d}"
            os.makedirs(save_dir, exist_ok=True)

            obs = {
                "action": actions, 
                "camera": images, 
                "mask": mask, 
                "instruction": task, 
            }
            torch.save(obs, f"{save_dir}/obs.pt")

        print("\t-> Complete converting.")
        pass

    def record_config(self, cfg, output_root):
        texts = json.dumps(cfg, indent=2, ensure_ascii=False)
        with open(f"{output_root}/config.txt", "w", encoding="utf-8") as f:
            f.write(texts)
        print(f"Save config to {output_root}/config.txt")
        pass

    def load_action_datas(self, path_list):
        all_data = []
        for path in path_list:
            joint_data = torch.load(f"{path}/obs.pt")["action"]
            all_data.append(joint_data.cpu().numpy())

        return all_data

    def plot_all_episodes(self, top_root, save_root):
        print("Plot action data of all episodes.")
        paths = sorted(glob(f"{top_root}/episode--*"))
        datas = self.load_action_datas(paths)

        fig, ax = plt.subplots(datas[0].shape[-1], 1, figsize=(12, 8))

        for data in tqdm(datas):
            for i in range(data.shape[-1]):
                ax[i].plot(data[:, i], color="C0")
                ax[i].set_ylim([-1.1, 1.1])
        fig.savefig(f"{save_root}/joint_plot.png", dpi=300, bbox_inches="tight")
        print(f"\t-> Save figure to '{save_root}/joint_plot.png'")
        pass

    def convert(self, ):
        episode_map, episode_list = self.generate_episode_group(self.dataset)
        self._convert_episode(self.dataset, episode_map, episode_list)
        self.record_config(self.cfg, self.output_root_meta)
        self.plot_all_episodes(self.output_root, self.output_root_meta)
        pass

    def _save_episode_gif(self, obs, cam_name, save_path, fps=10):
        imgs = obs["camera"][cam_name]  # (T, C, H, W)

        frames = []
        for i in range(len(imgs)):
            img = imgs[i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)

            img = (img * 255.0).astype(np.uint8)
            frames.append(img)

        imageio.mimsave(save_path, frames, fps=fps, loop=0)

    def _save_gif(self, root):
        print(f"Save .gif files.")
        cam_list = None
        epi_list = sorted(glob(f"{root}/episode--*"))
        for epi in tqdm(epi_list):
            obs = torch.load(f"{epi}/obs.pt", map_location="cpu")

            if cam_list is None:
                cam_list = obs["camera"].keys()
            
            for cam_name in cam_list:
                save_path = os.path.join(epi, f"{cam_name}.gif")
                self._save_episode_gif(obs, cam_name, save_path)
        print("\t-> Completed.")
        pass

    def save_gif(self, ):
        self._save_gif(root=self.output_root)
        pass


