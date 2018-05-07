import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from scipy.stats import multivariate_normal

# Adapted from https://github.com/mrzhu-cool/pix2pix-pytorch/blob/master/networks.py

class SLLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor, use_lsgan=True):
        super(SLLoss, self).__init__()

        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_weighting_tensor(self, pred, coords):
        weightings = []
        for coord in coords:
            center = coord[:2].cpu().data.numpy() / np.array([730/32, 530/32]) 
            half_length = coord[2:].cpu().data.numpy() / np.array([730/32, 530/32])


            m = np.mgrid[0:pred.shape[2]:1, 0:pred.shape[3]:1]
            m = m.T
          
            cov = [[half_length[0], 0],[0, half_length[1]]]
            weightings.append(multivariate_normal.pdf(m, mean=center, cov=cov).astype(float))

        
        target_tensor = torch.from_numpy(np.array(weightings))
        target_tensor = Variable(target_tensor, requires_grad=False)

        return target_tensor

    def apply_weightings(self, weightings, imgs):
        # Applys weighting along all channels of all images
        split = torch.unbind(imgs, dim=1)
        weighted_splits = [weightings * i for i in split]
        weighted_imgs = torch.stack(weighted_splits, dim=1)
        return weighted_imgs

    # pred and label are shape [batch, height, width, channels]
    # coords are of shape [batch, 4]
    def __call__(self, pred, label, coords):
        weightings = self.get_weighting_tensor(pred, coords).cuda().float()
        pred_weighted = self.apply_weightings(weightings, pred)
        label_weighted = self.apply_weightings(weightings, label)
        return self.loss(pred_weighted, label_weighted)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda())


def initialize_weights(m):
    classname = m.__class__.__name__
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
    G = Generator(input_channels, output_channels, num_filters, num_downsample, num_blocks, norm_fn=nn.BatchNorm2d, use_dropout=True, gpu_ids=gpu_ids)
    if use_gpu:
        G.cuda(gpu_ids[0])
    G.apply(initialize_weights)
    return G




class Generator(nn.Module):

    def __init__(self, input_channels, output_channels, num_filters, num_downsample, num_blocks, norm_fn=nn.BatchNorm2d, use_dropout=True, gpu_ids=[]):

        super(Generator, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = self.build_resnet_generator(num_filters, use_dropout, input_channels, output_channels, num_downsample, num_blocks, norm_fn)

    # Constructs network using resnet blocks.
    def build_resnet_generator(self, num_filters=64, use_dropout=True, input_channels=8, output_channels=4, num_downsample=2, num_blocks=6, norm_fn=nn.BatchNorm2d):
        layers = []
        layers = [nn.Conv2d(input_channels, num_filters, kernel_size=7, padding=3), norm_fn(num_filters, affine=True), nn.ReLU(True)]

        # Decoder
        for i in range(num_downsample):
            k = 2**i # 1, 2, 4, 8, ...
            layers += [nn.Conv2d(k*num_filters, k*num_filters*2, kernel_size=3, stride=2, padding=1), norm_fn(k*num_filters*2, affine=True), nn.ReLU(True)]

        # Mini-resnet
        k = 2**num_downsample
        for i in range(num_blocks):
            layers += [ResnetBlock(k*num_filters, norm_fn=norm_fn, use_dropout=use_dropout)]

        for i in range(num_downsample):
            k = 2**(num_downsample - i)
            layers += [nn.ConvTranspose2d(k*num_filters, int(k*num_filters/2), kernel_size=3, stride=2, padding=1, output_padding=1), 
                      norm_fn(int(k*num_filters/2), affine=True), nn.ReLU(True)]

        layers += [nn.Conv2d(num_filters, output_channels, kernel_size=7, padding=3)]
        layers += [nn.Tanh()]

        return nn.Sequential(*layers)
        
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_fn, use_dropout):
        super(ResnetBlock, self).__init__()
        self.resnet_block = self.build_conv_block(dim, use_dropout=use_dropout, norm_fn=norm_fn)

    def build_conv_block(self, dim, use_dropout=True, norm_fn=nn.BatchNorm2d):
        layers = []
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_fn(dim, affine=True)]
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        layers += [nn.ReLU(inplace=True), nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_fn(dim, affine=True)]

        return nn.Sequential(*layers)

    def forward(self, x):
        out = x + self.resnet_block(x)
        return out


def build_discriminator(input_channels, num_filters, norm_fn=nn.BatchNorm2d, use_sigmoid=True, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    netD = Discriminator(input_channels, num_filters, num_layers=3, norm_fn=norm_fn, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(initialize_weights)
    return netD

class Discriminator(nn.Module):
    def __init__(self, input_channels, num_filters=64, num_layers=3, norm_fn=nn.BatchNorm2d, use_sigmoid=True, gpu_ids=[]):
        super(Discriminator, self).__init__()
        self.gpu_ids = gpu_ids

        layers = [nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, True)]

        mult = 1
        mult_prev = mult
        for n in range(1, num_layers):
            mult_prev = mult
            mult = min(2**n, 8)
            layers += [nn.Conv2d(num_filters * mult_prev, num_filters * mult, kernel_size=3, stride=2, padding=1),
                         norm_fn(num_filters * mult, affine=True), nn.LeakyReLU(0.2, True)]

        layers += [nn.Conv2d(num_filters * mult, num_filters * mult, kernel_size=3, stride=1, padding=1),
                         norm_fn(num_filters * mult, affine=True), nn.LeakyReLU(0.2, True)]

        layers += [nn.Conv2d(num_filters * mult, 1, kernel_size=3, stride=1, padding=1)]

        # more layers here?

        if use_sigmoid:
            layers += [nn.Sigmoid()]

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        if len(self.gpu_ids) > 0 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)






def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)