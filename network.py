import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# Adapted from https://github.com/mrzhu-cool/pix2pix-pytorch/blob/master/networks.py

def initialize_weights():
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, 0.01)
            m.bias.data.zero_()

def build_generator(input_channels, output_channels, num_filters, num_downsample, num_blocks, norm_fn=nn.BatchNorm2d, use_dropout=True, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    G = Generator(input_channels, output_channels, num_filters, num_downsample, num_blocks, norm_fn=nn.BatchNorm2d, use_dropout=True, gpu_ids=[])
    if use_gpu:
        G.cuda(gpu_ids[0])
    G.apply(initialize_weights)
    return G

class Generator(nn.Module):

    def __init__(self, input_channels, output_channels, num_filters, num_downsample, num_blocks, norm_fn=nn.BatchNorm2d, use_dropout=True, gpu_ids=[]):

        super(Generator, self).__init__()
        self.gpu_ids = gpu_ids

        self.model = build_resnet_generator(num_filters, use_dropout, input_channels, out_channels, num_downsample, num_blocks, norm_fn)

    # Constructs network using resnet blocks.
    def build_resnet_generator(num_filters=64, use_dropout=True, input_channels=4, output_channels=3, num_downsample=2, num_blocks=6, norm_fn=nn.BatchNorm2d):
        layers = []
        layers = [nn.Conv2d(input_channels, num_filters, kernel_size=7, padding=3), norm_fn(num_filters, affine=True), nn.ReLU(affine=True)]

        # Decoder
        for i in range(num_downsample):
            k = 2**i # 1, 2, 4, 8, ...
            layers += [nn.Conv2d(k*num_filters, k*num_filters*2, kernel_size=3, stride=2, padding=1), norm_layer(k*num_filters*2, affine=True), nn.ReLU(affine=True)]

        # Mini-resnet
        k = 2**num_downsample
        for i in range(num_blocks):
            layers += [ResnetBlock(k*num_filters, norm_fn=norm_fn, use_dropout=use_dropout)]

        for i in range(num_downsample):
            k = 2**(num_downsample - i):
            layers += [nn.ConvTranspose2d(k*num_filters, int(k*num_filters/2), kernel_size=3, stride=2, padding=1, output_padding=1), 
                      norm_fn(int(k*num_filters/2), affine=True), nn.ReLU(affine=True)]

        layers += [nn.Conv2d(num_filters, output_channels=3, kernel_size=7, padding=3)]
        layers += [nn.Tanh()]

        self.model = nn.Sequential(*layers)
        
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_fn, use_dropout):
        super(ResnetBlock, self).__init__()
        self.resnet_block = self.build_conv_block(dim, use_dropout=use_dropout, norm_fn=norm_fn)

    def build_conv_block(dim, use_dropout=True, norm_fn=nn.BatchNorm2d):
        layers = []
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_fn(dim, affine=True)]
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        layers += [nn.ReLU(inplace=True), nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_fn(dim, affine=True)]
        return layers

    def forward(self, x):
        out = x + self.resnet_block(x)
        return out