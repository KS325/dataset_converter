
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm
import pandas as pd
import av
import torch
import numpy as np
import json

class LerobotDataset2Pt():
    def __init__(self, config):
        self.cfg = config
        pass

    def convert_dataset(self):
        dataset_root = Path(self.cfg.dataset.input.root)
        self.output_root = Path(os.path.join(self.cfg.dataset.output.root, self.cfg.dataset.input.data_name))
        os.makedirs(self.output_root, exist_ok=False)

        parquet_dir = Path(os.path.join(dataset_root, self.cfg.dataset.input.data_name, self.cfg.dataset.input.extra_path_action))
        videos_top_dir = Path(os.path.join(dataset_root, self.cfg.dataset.input.data_name, self.cfg.dataset.input.extra_path_image))
        instruction_path = Path(os.path.join(dataset_root, self.cfg.dataset.input.data_name, self.cfg.dataset.input.extra_path_meta))

        print("Loading parquet...")
        df = self.load_all_parquets(parquet_dir)

        instruction_map = self.build_instruction_map(instruction_path, self.cfg.dataset.instructions.start, len(df))

        print("Bulding video iterators...")
        cam_iters, cam_files = self.build_camera_iterators(videos_top_dir)

        epi_length = self.build_episode_length(df)
        episode_list = sorted(epi_length.keys())

        print(f"Start converting {len(episode_list)} episodes...")
        for epi in tqdm(episode_list):
            epi_len = epi_length[epi]

            action = self.extract_episode_action(df, epi)

            images = {}

            for cam_name, cam_iter in cam_iters.items():
                img = self.extract_episode_image(
                    cam_iter,
                    epi_len,
                )
                norm_cam_name = self.normalize_camera_name(cam_name)
                images[norm_cam_name] = img

            mask = torch.ones(epi_len, dtype=torch.bool)

            save_dir = f"{self.output_root}/episode-{epi}"
            os.makedirs(save_dir, exist_ok=True)

            obs = {
                "action": action, 
                "camera": images, 
                "mask": mask, 
                "instruction": instruction_map[epi], 
            }
            torch.save(obs, f"{save_dir}/obs.pt")
            # torch.save(action, f"{save_dir}/proprio.pt")
            # for cam_name, img in images.items():
            #     torch.save(img, f"{save_dir}/{cam_name}.pt")

        print(" --> Complete converting.")

        data_structure = {
            "action_max": convert_to_jsonable(self.action_max), 
            "action_min": convert_to_jsonable(self.action_min), 
            "obs": convert_to_jsonable(obs), 
        }
        with open(f"{self.meta_root}/data_structure.txt", "w") as f:
            json.dump(data_structure, f, indent=2, ensure_ascii=False)

        print(f"Save data structure to {self.meta_root}/data_structure.txt!")

    def extract_episode_image(self, frame_iter, epi_len):
        frames = []
        for _ in range(epi_len):
            try:
                frame = next(frame_iter)
            except StopIteration:
                raise RuntimeError("Video frame が不足しています. ")

            frames.append(frame)
        imgs = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()

        if self.cfg.preprocessing.image.normalize:
            imgs = imgs / 255.0

        if self.cfg.preprocessing.image.resize is not None:
            imgs = torch.nn.functional.interpolate(
                imgs, 
                size=self.cfg.preprocessing.image.resize, 
                mode="bilinear", 
                align_corners=False
            )

        return imgs

    def extract_episode_action(self, df: pd.DataFrame, epi: int):
        epi_df = (
            df[df["episode_index"] == epi]
            .sort_values("frame_index")
        )
        action = torch.from_numpy(np.stack(epi_df["action"].to_numpy())).float()

        if self.cfg.preprocessing.action.normalize:
            action = self.normalize_action(action)

        return action

    def build_episode_length(self, df: pd.DataFrame):
        epi_length = (
            df.groupby("episode_index")
            .size()
            .to_dict()
        )
        return epi_length

    def load_all_parquets(self, parquet_dir: Path):
        parquet_files = sorted(parquet_dir.glob("*.parquet"))
        dfs = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        self.compute_action_stats(df)

        print(" --> Complete loading.")
        return df

    def normalize_action(self, action):
        return 2 * (action - self.action_min) / (self.action_max - self.action_min + 1.0e-6) - 1

    def compute_action_stats(self, df):
        actions = np.stack(df["action"].to_numpy())
        self.action_min = torch.from_numpy(actions.min(axis=0)).float()
        self.action_max = torch.from_numpy(actions.max(axis=0) + 1e-6).float()

        self.meta_root = f"{self.output_root}/.meta"
        os.makedirs(self.meta_root, exist_ok=True)
        torch.save(self.action_min, f"{self.meta_root}/action_min.pt")
        torch.save(self.action_max, f"{self.meta_root}/action_max.pt")

    def build_camera_iterators(self, videos_root: Path):

        cam_dirs = self.discover_camera_video_dirs(videos_root)

        cam_iters = {}
        cam_files = {}

        for cam_name, cam_dir in cam_dirs.items():

            files = self.collect_camera_video_files(cam_dir)

            cam_files[cam_name] = files
            cam_iters[cam_name] = self.make_frame_iterator(files)

        print(" --> Complete building iterators.")
        return cam_iters, cam_files

    def make_frame_iterator(self, video_files: list):
        for vf in video_files:
            container = av.open(str(vf))
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")
            container.close()

    def collect_camera_video_files(self, cam_dir: Path):
        video_files = []

        chunk_dirs = sorted(cam_dir.glob("chunk-*"))

        for cd in chunk_dirs:
            video_files.extend(sorted(cd.glob("*.mp4")))

        return video_files

    def discover_camera_video_dirs(self, videos_root: Path):
        cam_dirs = {}

        for d in sorted(videos_root.iterdir()):
            if d.is_dir():
                cam_dirs[d.name] = d

        return cam_dirs

    def normalize_camera_name(self, name: str):
        return name.replace("observation.", "")

    def build_instruction_map(self, instructions_path: Path, task_start_epi: list, epi_num: int):
        task_map = self.load_task_instruction(instructions_path=instructions_path)
        task_indices = sorted(task_map.keys())
        assert len(task_start_epi) == len(task_indices)

        instruction_map = {}
        for epi_idx in range(epi_num):
            task_i = max(
                i for i, start in enumerate(task_start_epi)
                if start <= epi_idx
            )
            task_idx = task_indices[task_i]
            instruction_map[epi_idx] = task_map[task_idx]

        return instruction_map

    def load_task_instruction(self, instructions_path: Path):
        df_ins = pd.read_parquet(instructions_path)

        # task_index → instruction(index)
        task_map = dict(zip(df_ins["task_index"], df_ins.index))

        return task_map


    # def load_task_instruction(self, instructions_path: Path):
    #     df_ins = pd.read_parquet(instructions_path)
    #     task_map = {
    #         row["task_index"]: idx
    #         for idx, row in df_ins.reset_index().iterrows()
    #     }
    #     return task_map

    # def collect_video_files(self, video_dir: Path):
    #     video_files = sorted(video_dir.glob("*.mp4"))
    #     return video_files

def convert_to_jsonable(obj):

    if isinstance(obj, torch.Tensor):
        return tensor_to_jsonable(obj)

    elif isinstance(obj, dict):
        return {k: convert_to_jsonable(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [convert_to_jsonable(v) for v in obj]

    else:
        return obj

def tensor_to_jsonable(t):
    if isinstance(t, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(t.shape),
            "dtype": str(t.dtype),
        }
    else:
        return {
            "type": "str", 
        }
