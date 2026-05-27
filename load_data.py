
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def load_dataset(dataset_name):
    dataset = LeRobotDataset(
        repo_id=f"KS325/{dataset_name}",
        root=f"/home/sato/data/datasets/KS325/{dataset_name}"
    )

"""
merge dataset
lerobot-edit-dataset --new_repo_id KS325/grab-doll --operation.type merge --operation.repo_ids "['KS325/grab-doll-upper','KS325/grab-doll-lower']" --push_to_hub true

lerobot-edit-dataset --new_repo_id KS325/open-drawer-all-r1 --operation.type merge --operation.repo_ids "['KS325/open-upper-drawer','KS325/open-lower-drawer-r1']" --push_to_hub true

lerobot-edit-dataset --new_repo_id KS325/open-grab-r1 --operation.type merge --operation.repo_ids "['KS325/open-drawer-all-r1','KS325/place-doll-all-r1']" --push_to_hub true

lerobot-edit-dataset --new_repo_id KS325/open-drawer-all-r2 --operation.type merge --operation.repo_ids "['KS325/open-upper-drawer-r1','KS325/open-lower-drawer-r1']" --push_to_hub true

lerobot-edit-dataset --new_repo_id KS325/open-grab-r2 --operation.type merge --operation.repo_ids "['KS325/open-drawer-all-r2','KS325/place-doll-all-r1']" --push_to_hub true

lerobot-edit-dataset --new_repo_id KS325/close-drawer-all-r1 --operation.type merge --operation.repo_ids "['KS325/close-upper-drawer-r1','KS325/close-lower-drawer-r1']" --push_to_hub true

lerobot-edit-dataset --new_repo_id KS325/skill-set-r1 --operation.type merge --operation.repo_ids "['KS325/open-drawer-all-r2','KS325/place-doll-all-r1', 'KS325/close-drawer-all-r1']" --push_to_hub true

lerobot-edit-dataset --new_repo_id KS325/skill-set-r1 --operation.type merge --operation.repo_ids "['KS325/open-drawer-all-r2','KS325/place-doll-upper-r1', 'KS325/place-doll-lower-r1', 'KS325/close-drawer-all-r1']" --push_to_hub true

lerobot-edit-dataset --new_repo_id KS325/place-doll-all-r1 --operation.type merge --operation.repo_ids "['KS325/place-doll-upper-r1','KS325/place-doll-lower-r1']" --push_to_hub true

lerobot-edit-dataset --new_repo_id KS325/skill-set --operation.type merge --operation.roots "['/home/sato/data/datasets/KS325/open-upper-drawer', '/home/sato/data/datasets/KS325/grab-doll-upper', '/home/sato/data/datasets/KS325/grab-doll-lower', '/home/sato/data/datasets/KS325/place-doll-all', '/home/sato/data/datasets/KS325/close-upper-drawer', '/home/sato/data/datasets/KS325/open-lower-drawer', '/home/sato/data/datasets/KS325/close-lower-drawer']" --operation.repo_ids "['', '', '', '', '', '', '']" --push_to_hub true
'/home/sato/data/datasets/KS325/grab-doll', 

lerobot-edit-dataset --new_repo_id KS325/skill-set --operation.type merge --operation.roots "['/home/sato/data/datasets/KS325/open-upper-drawer', '/home/sato/data/datasets/KS325/open-lower-drawer']" --operation.repo_ids "['', '']" --push_to_hub true



lerobot-edit-dataset --repo_id KS325/skill-set --operation.type split --operation.splits '{"train": 0.7, "test": 0.1, "val": 0.2}' --push_to_hub true 

# Merge train and validation splits back into one dataset
lerobot-edit-dataset --new_repo_id KS325/skill-set-splited --operation.type merge --operation.repo_ids "['KS325/skill-set_train', 'KS325/skill-set_val', 'KS325/skill-set_test']" --push_to_hub true



"""
import argparse

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
