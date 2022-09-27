#!/bin/bash

# To generate the samples
python scripts/txt2img.py --ddim_eta 0.0 --n_samples 1 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path /data/abhishek/projects/textual_inversion/logs/train2022-09-25T18-43-44_dress_init_word-cloth/checkpoints/embeddings_gs-6099.pt --ckpt_path /data/abhishek/projects/textual_inversion/models/ldm/text2img-large/model.ckpt --prompt "pink crop top  on * in the style of *"