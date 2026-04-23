
import torch
from transformers import SiglipProcessor, SiglipModel, SiglipImageProcessor
import torch.nn as nn
import os
from pathlib import Path
from tqdm import tqdm

class Encoder():
    def __init__(self, config, device):
        self.cfg = config
        self.device = device
        self.image_encoder = SigLIP2ImageEncoder(config, device).to(device)
        self.image_encoder.eval()
        self.output_root, self.output_root_meta = self._generate_output_root()
        pass

    def _generate_output_root(self, ):
        output_root = Path(os.path.join(self.cfg.dataset.output.root, self.cfg.dataset.input.data_name))
        output_root_meta = Path(os.path.join(output_root, ".meta"))
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(output_root_meta, exist_ok=True)

        return output_root, output_root_meta

    def add_feature(self, ):
        print(f"Start encoding images w/ {self.cfg.preprocess.encode.image_encoder}.")
        episode_dirs = sorted([
            d for d in os.listdir(self.output_root)
            if d.startswith("episode--")
        ])

        for epi_dir in tqdm(episode_dirs):
            obs_path = os.path.join(self.output_root, epi_dir, "obs.pt")

            obs = torch.load(obs_path)

            # if "feature" in obs:
            #     print(f"Skip {epi_dir} (already has feature)")
            #     continue

            images = obs["camera"]
            features = {}

            with torch.no_grad():
                for cam_name, imgs in images.items():
                    imgs = imgs.to(self.device)

                    # chunk_size = 32
                    # feat_list = []

                    # for i in range(0, imgs.shape[0], chunk_size):
                    #     chunk = imgs[i:i+chunk_size]
                    #     feat = self.image_encoder(chunk)   # (t, D)
                    #     feat_list.append(feat.cpu())

                    # features[cam_name] = torch.cat(feat_list, dim=0)
                    feat = self.image_encoder(imgs).cpu()
                    features[cam_name] = feat

            obs["feature"] = features
            torch.save(obs, obs_path)
        print("\t -> Complete encoding.")
        pass

class SigLIP2ImageEncoder(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        model_name = str(cfg.preprocess.encode.image_encoder)
        self.processor = SiglipImageProcessor.from_pretrained(model_name)
        # self.last_processor = SiglipProcessor.from_pretrained(model_name)
        self.image_encoder = SiglipModel.from_pretrained(model_name).to(device)
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        self.image_encoder.eval()
        self.device = device

    def forward(self, images) -> torch.Tensor:
        if images.dim() == 4:
            images = images.unsqueeze(0)  # (1, T, C, H, W)
        B, T, C, H, W = images.shape
        images = images.view(B*T, C, H, W)
        # print(images.max(), images.min())

        siglip_inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        siglip_inputs = {k: v.to(self.device) for k, v in siglip_inputs.items()}

        with torch.no_grad():
            image_embed = self.image_encoder.get_image_features(**siglip_inputs)
        image_embed = image_embed.view(B, T, -1).squeeze(0)

        return image_embed
