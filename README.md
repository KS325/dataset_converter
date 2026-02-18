# dataset_converter
## 概要
ネット上に公開されているデータセットを, 自分のローカル環境に適したフォルダ構造で保存しなおすためのリポジトリ. 
実装済み一覧
 - Lerobot Dataset v3.0形式で保存されたデータをエピソード毎に切り出して個々のフォルダに保存しなおす

## src/lerobot/dataset_3_0
Lerobot Dataset v3.0形式のデータをエピソード毎に保存しなおすための実装. 
### Lerobot Dataset v3.0 のフォルダ構造
ローカルにダウンロードした際に, 複数のエピソードが繋がった状態 (concatされた状態)で保存されている. 
```text
user_name/
 └ dataset_name/
    ├ .cache/
    ├ data/
      └ chunk-000/
          ├ file-000.parquet
          ├ file-001.parquet
            ...
    ├ meta/
    ├ videos/
      └ observation.camera1/
          └ chunk-000/
              ├ file-000.mp4
              ├ file-001.mp4
                ...
```
### 目標とするフォルダ構造
関節角度, 画像それぞれをエピソード毎に分けた後.ptファイルに保存する.
```text
user_name/
 └ dataset_name/
    ├ .meta
       ├ action_min.pt
       ├ action_max.pt
       ├ config.txt
       └ data_structure.txt
    ├ episode-0/
       └ obs.pt
    ├ episode-1/
       └ obs.pt
```
### .meta
各ファイルの中身は以下の通り. 
 - action_min.pt (`torch.Tensor(]dim, ])`) -> 取得した全エピソード内での関節角度の最小値 (各関節ごと)
 - action_max.pt (`torch.Tensor([dim, ])`) -> 取得した全エピソード内での関節角度の最大値 (各関節ごと)
 - config.txt -> 変換時の設定を`json.dump()`して保存.
 - data_structure.txt -> obs.pt の中身の形状を保存. 
### episode-*n*/obs.pt
obsの中身と型は以下の形式で保存される. 
```python
obs = {
    "action": torch.Tensor([num_steps, dim]), 
    "camera"{
        images.camera1 : torch.Tensor([num_steps, C, H, W]),
        images.camera2 : torch.Tensor([num_steps, C, H, W]),
        ...
    }, 
    "mask": torch.Tensor([num_steps, ]), 
    "instruction": str, 
}
```
 - action : 関節角度 (-1 ~ 1への正規化が可能)
 - camera : 各カメラの画像 (0 ~ 1への正規化, 画像のリサイズが可能)
 - mask : [1, 1, 1, ..., 1] (`len = num_steps`)
 - instruction : データ取得時の言語指示
### config
`./conf/config.yaml`を作成し, 変換に関する設定を書き込む必要がある
```yaml
dataset:
  input:
    root: "{path_to_local_dataset}"          # ex) ~/.cache/huggingface
    data_name: "{user_name}/{dataset_name}"
    extra_path_action: "data/chunk-000"      # No need to rewrite, maybe...
    extra_path_image: "videos"               # No need to rewrite, maybe...
    extra_path_meta: "meta/tasks.parquet"    # No need to rewrite, maybe...

  output:
    root: "{path_to_output}"                 # ex) ~/dataset/pt_datasets (No need to include "user name" or "dataset name.")

  instructions:
    start: [0, 30]                           # If the data contains multiple tasks, record the episode that started each task in list format. "[0]" if not.

preprocessing:
  image:
    normalize: True                          # True : images /= 255.0
    resize: [48, 64]                         # resize shape. "False" if no resizing is needed.

  action:
    normalize: True                          # True : action is normalized to -1.0 ~ 1.0
```
### 使い方
 - 環境構築
```bash
git clone https://github.com/KS325/dataset_converter.git
cd dataset_converter
python -m venv .venv
source .venv/bin/activate
python -m pip install pandas pyarrow av torch tqdm python-box pyyaml
```
 - `./main.py`で以下を実行.
```python
import json
from box import Box
import yaml

from src.lerobot.dataset_3_0.convert_to_pt import LerobotDataset2Pt

def load_config():
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
```
