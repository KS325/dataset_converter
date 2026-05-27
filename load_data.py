
import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def load_dataset(dataset_name):
    dataset = LeRobotDataset(
        repo_id=f"KS325/{dataset_name}",
        root=f"/home/sato/data/datasets/KS325/{dataset_name}"
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str)
    return parser.parse_args()

def check_instruction(dataset_name):
    dataset = LeRobotDataset(
        repo_id="", 
        root=f"/home/sato/data/datasets/KS325/{dataset_name}"
    )
    print(f"type(dataset): {type(dataset)}")
    print(f"len: {len(dataset)}")
    print(f"type(dataset[0]): {type(dataset[0])}")
    print(f"keys: {dataset[0].keys()}")

if __name__ == "__main__":
    dataset_name = "skill-set-r1-train"
    # check_instruction(dataset_name)
    load_dataset(dataset_name)
