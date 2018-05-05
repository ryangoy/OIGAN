from __future__ import print_function
import argparse
import os

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from load_data import is_image_file, load_img, save_img
import numpy as np
from os import listdir
from os.path import join

# Testing settings
parser = argparse.ArgumentParser(description='OIGAN')
parser.add_argument('--model', type=str, default='checkpoint/sunrgbd/netG_model_epoch_90.pth', help='model file to use')
parser.add_argument('--cuda', type=bool, default='True', help='use cuda')
opt = parser.parse_args()
print(opt)


netG = torch.load(opt.model)

fg_image_dir = "data_processing/test/foreground/"
bg_image_dir = "data_processing/test/background/"
image_filenames = [x for x in listdir(fg_image_dir) if (is_image_file(x) and os.path.isfile(join(bg_image_dir, x)))]
image_filenames.sort()

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    fg_img = np.array(load_img(fg_image_dir + image_name))
    bg_img = np.array(load_img(bg_image_dir + image_name))
    img = np.concatenate([bg_img, fg_img], -1)
    img = transform(img)
    input = Variable(img, volatile=True).view(1, -1, 256, 256)

    if opt.cuda:
        netG = netG.cuda()
        input = input.cuda()

    out = netG(input)
    out = out.cpu()
    out_img = out.data[0]
    if not os.path.exists("result"):
        os.makedirs("result")
    save_img(out_img[:3], "result/{}".format(image_name))