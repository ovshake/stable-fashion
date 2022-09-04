from dataclasses import dataclass
from diffusers import UNet2DModel, UNet2DConditionModel
import torch
from torch import nn



model =  UNet2DConditionModel(sample_size=128, in_channels=3, out_channels=3, layers_per_block=2)
sample_image = torch.randn((1, 3, 128, 128))
print('Output shape:', model(sample_image, encoder_hidden_states=torch.randn((1, 2, 1280), dtype=torch.float) ,timestep=0)["sample"].shape)
