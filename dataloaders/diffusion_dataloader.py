import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
import numpy as np

class StableDataset(Dataset):
    def __init__(self, root_dir, transforms):
        # datapath /data/dataset/VITON-hD/cloth/
        self.img_dir = os.path.join(root_dir, "cloth")
        self.np_embedding_dir = os.path.join(root_dir, "clip_txt_embeddings")
        self.transforms = transforms
        self.paths = glob(os.path.join(self.img_dir, "*.jpg"))
        self.names = os.listdir(self.img_dir)
        self.names = [x.replace(".jpg", "") for x in self.names]


    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        img_path = os.path.join(self.img_dir, f"{name}.jpg")
        np_embedding_path = os.path.join(self.np_embedding_dir, f"{name}.np.gz")
        img = Image.open(img_path)
        img = self.transforms(img)
        np_embedding = np.loadtxt(np_embedding_path, dtype=np.dtype('float32'))
        np_embedding = torch.from_numpy(np_embedding).unsqueeze(0)
        return {"images": img, "np_embedding": np_embedding}