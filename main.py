import gdown
import numpy as np
from PIL import Image
import gdown
import os
import sys
import os.path as osp
import time
import shutil
from ACGPN.predict_pose import generate_pose_keypoints
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Parameters for Stable Fashion')
parser.add_argument('--prompt',
                    type=str,
                    help='please insert your prompt describing the style',
                    default='a pink shirt')

parser.add_argument('--pic',
                    type=str,
                    help='path to your full body pic, in jpg format',
                    default='textual_inversion/man.jpeg')

args = parser.parse_args()
mkdir_commands = [
    "mkdir -p ACGPN/Data_preprocessing/test_color",
    "mkdir -p ACGPN/Data_preprocessing/test_colormask",
    "mkdir -p ACGPN/Data_preprocessing/test_edge",
    "mkdir -p ACGPN/Data_preprocessing/test_img",
    "mkdir -p ACGPN/Data_preprocessing/test_label",
    "mkdir -p ACGPN/Data_preprocessing/test_mask",
    "mkdir -p ACGPN/Data_preprocessing/test_pose",
    "mkdir -p ACGPN/inputs",
    "mkdir -p ACGPN/inputs/img",
    "mkdir -p ACGPN/inputs/cloth",
]

for mkdir_command in mkdir_commands:
    subprocess.run(mkdir_command, shell=True)


if not osp.exists('ACGPN/pose/pose_iter_440000.caffemodel'):
    # os.chdir('pose/')
    subprocess.run('gdown --id 1hOHMFHEjhoJuLEQY0Ndurn5hfiA9mwko -O ACGPN/pose/')
    # os.chdir('..')

if not osp.exists('ACGPN/lip_final.pth'):
    url = 'https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH'
    output = 'ACGPN/lip_final.pth'
    gdown.download(url, output, quiet=False)


# os.chdir('ACGPN/U_2_Net/')
os.makedirs("ACGPN/U_2_Net/saved_models", exist_ok=True)
os.makedirs("ACGPN/U_2_Net/saved_models/u2net", exist_ok=True)
os.makedirs("ACGPN/U_2_Net/saved_models/u2netp", exist_ok=True)

if not osp.exists("ACGPN/U_2_Net/saved_models/u2netp/u2netp.pth"):
    subprocess.run("gdown --id 1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy -O ACGPN/U_2_Net/saved_models/u2netp/u2netp.pth")
if not osp.exists("ACGPN/U_2_Net/saved_models/u2net/u2net.pth"):
    subprocess.run("gdown --id 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ -O ACGPN/U_2_Net/saved_models/u2net/u2net.pth")


from  ACGPN.U_2_Net import u2net_load
from ACGPN.U_2_Net import u2net_run
u2net = u2net_load.model(model_name = 'u2netp')


os.makedirs("ACGPN/checkpoints", exist_ok=True)
if not osp.exists("ACGPN/checkpoints/label2city"):
    gdown.download('https://drive.google.com/uc?id=1UWT6esQIU_d4tUm8cjxDKMhB8joQbrFx',output='ACGPN/checkpoints/ACGPN_checkpoints.zip', quiet=False)
    subprocess.run('unzip ACGPN/checkpoints/ACGPN_checkpoints.zip -d ACGPN/checkpoints', shell=True)

# Insert Image of Cloth from LD


if osp.exists('textual_inversion/outputs/txt2img-samples'):
    shutil.rmtree('textual_inversion/outputs/txt2img-samples')

if not osp.exists('textual_inversion/models/ldm/text2img-large/model.ckpt'):
    os.makedirs('textual_inversion/models/ldm/text2img-large/', exist_ok=True)
    subprocess.run('wget -O textual_inversion/models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt', shell=True)

if not osp.exists('textual_inversion/finetuned_models/'):
    os.makedirs('textual_inversion/finetuned_models/', exist_ok=True)
    subprocess.run('gdown --id 1AsDkbfZnQUwTof_I2ajRxsTjuYyeVwra -O textual_inversion/finetuned_models/', shell=True)

if osp.exists('textual_inversion/outputs'):
    shutil.rmtree('textual_inversion/outputs')

subprocess.run(f"python textual_inversion/scripts/txt2img.py --ddim_eta 0.0 --n_samples 1 --outdir textual_inversion/outputs --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path textual_inversion/finetuned_models/embeddings_gs-6099.pt --ckpt_path textual_inversion/models/ldm/text2img-large/model.ckpt --prompt '{args.prompt} on * in the style of *' ", shell=True)

shutil.copy('textual_inversion/outputs/samples/0000.jpg', 'ACGPN/inputs/cloth/0000.jpg')
shutil.rmtree('textual_inversion/outputs/')


# Insert Image of Person
# subprocess.run('gdown --id 1CWpTgqKGuZwgR8sxMfg6qKuy3wCAoOjc -O textual_inversion/man.jpeg', shell=True)
# shutil.copy('textual_inversion/man.jpeg', 'ACGPN/inputs/img/0000.jpg')
shutil.copy(args.pic, 'ACGPN/inputs/img/0000.jpg')

cloth_name = '000001_1.png'
cloth_path = os.path.join('ACGPN/inputs/cloth', sorted(os.listdir('ACGPN/inputs/cloth'))[0])
cloth = Image.open(cloth_path)
cloth = cloth.resize((192, 256), Image.BICUBIC).convert('RGB')
cloth.save(os.path.join('ACGPN/Data_preprocessing/test_color', cloth_name))

u2net_run.infer(u2net, 'ACGPN/Data_preprocessing/test_color', 'ACGPN/Data_preprocessing/test_edge')


start_time = time.time()
img_name = '000001_0.png'
img_path = os.path.join('ACGPN/inputs/img', sorted(os.listdir('ACGPN/inputs/img'))[0])
img = Image.open(img_path)
img = img.resize((192,256), Image.BICUBIC)

img_path = os.path.join('ACGPN/Data_preprocessing/test_img', img_name)
img.save(img_path)
resize_time = time.time()
print('Resized image in {}s'.format(resize_time-start_time))

subprocess.run("python3 ACGPN/Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'ACGPN/lip_final.pth' --input-dir 'ACGPN/Data_preprocessing/test_img' --output-dir 'ACGPN/Data_preprocessing/test_label'", shell=True)
parse_time = time.time()
print('Parsing generated in {}s'.format(parse_time-resize_time))

pose_path = os.path.join('ACGPN/Data_preprocessing/test_pose', img_name.replace('.png', '_keypoints.json'))
generate_pose_keypoints(img_path, pose_path)
pose_time = time.time()
print('Pose map generated in {}s'.format(pose_time-parse_time))

with open('ACGPN/Data_preprocessing/test_pairs.txt','w') as f:
    f.write('000001_0.png 000001_1.png')

subprocess.run('python ACGPN/test.py', shell=True)
shutil.copy('ACGPN/results/test/try-on/000001_0.png', 'result.png')
