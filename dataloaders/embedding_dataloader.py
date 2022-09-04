import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob

class EmbeddingDataset(Dataset):
    def __init__(self, img_dir, transforms):
        # datapath /data/dataset/VITON-hD/cloth/
        self.img_dir = img_dir
        self.transforms = transforms
        self.paths = glob(os.path.join(self.img_dir, "*.jpg"))


    def __len__(self):
        return len(self.paths) # hardcoded to save time

    def __getitem__(self, index):
        img_path = self.paths[index]
        img_name = img_path.split("/")[-1].split('.')[0]
        img = Image.open(img_path)
        img = self.transforms(img)
        return img_name, img







