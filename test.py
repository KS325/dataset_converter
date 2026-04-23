
import torch
from glob import glob
from tqdm import tqdm

def main():
    root = "/home/sato/data/pt_datasets/KS325/skill-set"
    epi_list = sorted(glob(f"{root}/episode--*"))
    inst_list = {}

    for epi in tqdm(epi_list):
        inst = torch.load(f"{epi}/obs.pt")["instruction"]

        if inst not in inst_list.values():
            idx = epi.split("-")[-1]
            inst_list[idx] = inst

    print(inst_list)

if __name__ == "__main__":
    main()

