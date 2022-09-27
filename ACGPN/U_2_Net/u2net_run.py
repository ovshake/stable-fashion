import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
from ACGPN.U_2_Net.u2net_test import normPRED
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import warnings

from ACGPN.U_2_Net.data_loader import RescaleT
from ACGPN.U_2_Net.data_loader import ToTensor
from ACGPN.U_2_Net.data_loader import ToTensorLab
from ACGPN.U_2_Net.data_loader import SalObjDataset

warnings.filterwarnings("ignore")

def save_images(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BICUBIC)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    print('Saving output at {}'.format(os.path.join(d_dir, imidx+'.png')))
    imo.save(os.path.join(d_dir, imidx+'.png'))

def infer(
    net,
    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images'),
    prediction_dir = os.path.join(os.getcwd(), 'test_data', 'u2net' + '_results')
    ):


    img_name_list = glob.glob(image_dir + os.sep + '*')
    prediction_dir = prediction_dir + os.sep

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("Generating mask for:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_images(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7
