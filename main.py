
import json
from box import Box
import yaml

from src.lerobot.dataset_3_0.convert_to_pt import LerobotDataset2Pt

def load_config():
    # Config.yaml の呼び出し
    with open("./conf/config.yaml", "r") as yml:
        cfg = Box(yaml.safe_load(yml))
    return cfg

def record_config(cfg, path_to_output_dir):
    texts = json.dumps(cfg, indent=2, ensure_ascii=False)
    with open(f"{path_to_output_dir}/config.txt", "w", encoding="utf-8") as f:
        f.write(texts)
        print(f"Save config to {path_to_output_dir}/config.txt !")
    pass

def main():
    cfg = load_config()
    converter = LerobotDataset2Pt(cfg)
    converter.convert_dataset()
    record_config(cfg, converter.meta_root)

if __name__ == "__main__":
    main()
