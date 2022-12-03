from diffusers import StableDiffusionInpaintPipeline
import os

from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu
from networks import U2NET
import argparse
from enum import Enum
from rembg import remove

class Parts:
    UPPER = 1
    LOWER = 2

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Stable Fashion API, allows you to picture yourself in any cloth your imagination can think of!"
    )
    parser.add_argument('--image', type=str, required=True, help='path to image')
    parser.add_argument('--part', choices=['upper', 'lower'], default='upper', type=str)
    parser.add_argument('--resolution', choices=[256, 512, 1024, 2048], default=256, type=int)
    parser.add_argument('--prompt', type=str, default="A pink cloth")
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--rembg', action='store_true')
    parser.add_argument('--output', default='output.jpg', type=str)
    args, _ = parser.parse_known_args()
    return args


def load_u2net():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm_u2net_latest.pth")
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    return net

def change_bg_color(rgba_image, color):
    new_image = Image.new("RGBA", rgba_image.size, color)
    new_image.paste(rgba_image, (0, 0), rgba_image)
    return new_image.convert("RGB")


def load_inpainting_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
    return inpainting_pipeline
def process_image(args, inpainting_pipeline, net):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = args.image
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize((args.resolution, args.resolution))
    if args.rembg:
        img_with_green_bg = remove(img)
        img_with_green_bg = change_bg_color(img_with_green_bg, color="GREEN")
        img_with_green_bg = img_with_green_bg.convert("RGB")
    else:
        img_with_green_bg = img
    image_tensor = transform_rgb(img_with_green_bg)
    image_tensor = image_tensor.unsqueeze(0)
    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()
    mask_code = eval(f"Parts.{args.part.upper()}")
    mask = (output_arr == mask_code)
    output_arr[mask] = 1
    output_arr[~mask] = 0
    output_arr *= 255
    mask_PIL = Image.fromarray(output_arr.astype("uint8"), mode="L")
    clothed_image_from_pipeline = inpainting_pipeline(prompt=args.prompt,
                                                    image=img_with_green_bg,
                                                    mask_image=mask_PIL,
                                                    width=args.resolution,
                                                    height=args.resolution,
                                                    guidance_scale=args.guidance_scale,
                                                    num_inference_steps=args.num_steps).images[0]
    clothed_image_from_pipeline = remove(clothed_image_from_pipeline)
    clothed_image_from_pipeline = change_bg_color(clothed_image_from_pipeline, "WHITE")
    return clothed_image_from_pipeline.convert("RGB")
if __name__ == '__main__':
    args = parse_arguments()
    net = load_u2net()
    inpainting_pipeline = load_inpainting_pipeline()
    result_image = process_image(args, inpainting_pipeline, net)
    result_image.save(args.output)
