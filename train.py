from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks import build_generator, build_discriminator, GANLoss, print_network
from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--input_channels', type=int, default=4, help='input image channels')
parser.add_argument('--output_channels', type=int, default=3, help='output image channels')
parser.add_argument('--num_gen_filters', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--num_dis_filters', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--num_gen_ds', type=int, default=64, help='number of downsample operations in generator')
parser.add_argument('--num_dis_ds', type=int, default=64, help='number of downsample operations in discriminator')
parser.add_argument('--num_gen_blocks', type=int, default=64, help='number of resnet blocks in generator')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

# Load data



# Make the model
G = build_generator(opt.input_channels, opt.output_channels, opt.num_gen_filters, opt.num_gen_ds, opt.num_gen_blocks, 
                    norm_fn=nn.BatchNorm2d, use_dropout=False, gpu_ids=[0])



# Set up loss




# Set up optimizer




# Train