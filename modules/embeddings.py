import torch
from torch import nn
from dataloaders.embedding_dataloader import EmbeddingDataset
import clip
from clip_decoder import CLIPDecoder
from torch.utils.data import DataLoader
import os
import numpy as np
from joblib import Parallel, delayed
def dump_embedding(np_embedding, img_name):
    dump_dir = "/data/dataset/VITON-hD/train/clip_txt_embeddings"
    dump_file_name = os.path.join(dump_dir, f"{img_name}.np.gz")
    np.savetxt(dump_file_name, np_embedding)

def train(model, dataloader):
    text = ["A picture of cloth on white background"]
    for imgs_name, imgs in dataloader:
        text_embeddings = model.get_embedding(imgs.cuda(non_blocking=True), text)
        Parallel(n_jobs=2)(delayed(dump_embedding)(np_embedding, img_name) for np_embedding, img_name in zip(text_embeddings, imgs_name))



def main():
    device = "cuda"
    model, preprocess = clip.load("ViT-L/14", device=device)
    clip_decoder = CLIPDecoder(model, epochs=1000)
    dataset = EmbeddingDataset("/data/dataset/VITON-hD/train/cloth/", transforms=preprocess)
    dataloader = DataLoader(dataset,
                        batch_size=128,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True)
    train(clip_decoder, dataloader)


if __name__ == '__main__':
    main()