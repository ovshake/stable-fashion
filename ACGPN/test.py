import time
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
#from tensorboardX import SummaryWriter
import cv2
#writer = SummaryWriter('runs/G1G2')
SIZE = 320
NC = 14


def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256, 192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256, 192)

    return label_batch


def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], NC))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label


def complete_compose(img, mask, label):
    label = label.cpu().numpy()
    M_f = label > 0
    M_f = M_f.astype(np.int)
    M_f = torch.FloatTensor(M_f).cuda()
    masked_img = img*(1-mask)
    M_c = (1-mask.cuda())*M_f
    M_c = M_c+torch.zeros(img.shape).cuda()  # broadcasting
    return masked_img, M_c, M_f


def compose(label, mask, color_mask, edge, color, noise):
    masked_label = label*(1-mask)
    masked_edge = mask*edge
    masked_color_strokes = mask*(1-color_mask)*color
    masked_noise = mask*noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise


def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor((old_label.cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor((old_label.cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor((old_label.cpu().numpy() == 7).astype(np.int))
    label = label*(1-arm1)+arm1*4
    label = label*(1-arm2)+arm2*4
    label = label*(1-noise)+noise*4
    return label


def main():
    os.makedirs('ACGPN/sample', exist_ok=True)
    opt = TestOptions().parse()

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# Inference images = %d' % dataset_size)

    model = create_model(opt)

    for i, data in enumerate(dataset):

        # add gaussian noise channel
        # wash the label
        t_mask = torch.FloatTensor(
            (data['label'].cpu().numpy() == 7).astype(np.float))
        #
        # data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        mask_clothes = torch.FloatTensor(
            (data['label'].cpu().numpy() == 4).astype(np.int))
        mask_fore = torch.FloatTensor(
            (data['label'].cpu().numpy() > 0).astype(np.int))
        img_fore = data['image'] * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(data['label'])

        ############## Forward Pass ######################
        fake_image, warped_cloth, refined_cloth = model(Variable(data['label'].cuda()), Variable(data['edge'].cuda()), Variable(img_fore.cuda()), Variable(
            mask_clothes.cuda()), Variable(data['color'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['image'].cuda()), Variable(data['pose'].cuda()), Variable(data['image'].cuda()), Variable(mask_fore.cuda()))

        # make output folders
        output_dir = os.path.join(opt.results_dir, opt.phase)
        fake_image_dir = os.path.join(output_dir, 'try-on')
        os.makedirs(fake_image_dir, exist_ok=True)
        warped_cloth_dir = os.path.join(output_dir, 'warped_cloth')
        os.makedirs(warped_cloth_dir, exist_ok=True)
        refined_cloth_dir = os.path.join(output_dir, 'refined_cloth')
        os.makedirs(refined_cloth_dir, exist_ok=True)

        # save output
        for j in range(opt.batchSize):
            print("Saving", data['name'][j])
            util.save_tensor_as_image(fake_image[j],
                                      os.path.join(fake_image_dir, data['name'][j]))
            util.save_tensor_as_image(warped_cloth[j],
                                      os.path.join(warped_cloth_dir, data['name'][j]))
            util.save_tensor_as_image(refined_cloth[j],
                                      os.path.join(refined_cloth_dir, data['name'][j]))


if __name__ == '__main__':
    main()
