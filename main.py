
## .venv を activateして実行
## frame 単位での同期がとれている

import json
from box import Box
import yaml
import torch
import argparse

def load_config():
    # Config.yaml の呼び出し
    with open("./conf/config.yaml", "r") as yml:
        cfg = Box(yaml.safe_load(yml))
    return cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str)
    return parser.parse_args()

def override_cfg(cfg, args):
    if args.data_name is not None:
        cfg.dataset.input.data_name = args.data_name
    return cfg

def main():
    args = parse_args()
    cfg = load_config()
    cfg = override_cfg(cfg, args)
    if cfg.runtype.convert:
        from src.convert_lerobot_dataset.dataset_3_0.convert_to_pt import LerobotDataset2Pt

        converter = LerobotDataset2Pt(cfg)
        converter.convert()

        if cfg.runtype.save_gif:
            converter.save_gif()

    if cfg.runtype.encode.image:
        from src.convert_lerobot_dataset.dataset_3_0.image_encoder import Encoder

        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print(device)
        encoder = Encoder(cfg, device)
        encoder.add_feature()
        pass

if __name__ == "__main__":
    main()

