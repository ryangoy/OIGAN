from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks import build_generator, build_discriminator, GANLoss, SLLoss, print_network

from load_data import save_img

import torch.backends.cudnn as cudnn

from load_data import get_training_set, get_test_set
import numpy as np

from tensorboardX import SummaryWriter
writer = SummaryWriter()


# Training settings
parser = argparse.ArgumentParser(description='OIGAN PyTorch implementation')
parser.add_argument('--dataset', default='sunrgbd', help='type of dataset')
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--input_channels', type=int, default=8, help='input image channels')
parser.add_argument('--output_channels', type=int, default=4, help='output image channels')
parser.add_argument('--num_gen_filters', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--num_dis_filters', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--num_gen_ds', type=int, default=3, help='number of downsample operations in generator')
parser.add_argument('--num_dis_ds', type=int, default=3, help='number of downsample operations in discriminator')
parser.add_argument('--num_gen_blocks', type=int, default=8, help='number of resnet blocks in generator')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# Optimizations
parser.add_argument('--lamb', type=int, default=10, help='DEPRECIATED: weight on L1 term in objective')
parser.add_argument('--l1_bonus', type=int, default=1, help='weight on L1 term in objective')
parser.add_argument('--sl_bonus', type=int, default=1e9, help='weight on SLterm in objective')
parser.add_argument('--gan_bonus', type=int, default=1, help='weight on gan term in objective')
parser.add_argument('--include_depth', type=int, default=1, help='1: include depth. 0: No')
parser.add_argument('--step_ratio', type=int, default=1, help='Number of G steps wrt D steps')


# Restore
parser.add_argument('--restore_G', type=str, default=None, help='generative model file to use')
parser.add_argument('--restore_D', type=str, default=None, help='discriminative model file to use')



opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)


root_path = "data_processing/"
train_set = get_training_set(root_path)
test_set = get_test_set(root_path)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

def add_images(imgs, writer, name, num_steps):
    imgs = imgs.clone()
    _, C, _, _ = imgs.shape
    img = imgs[0].cpu()
    if C == 3:
        img = unorm(img)
        writer.add_image(name, img, num_steps)

    if C == 4:
        rgb, d = torch.split(img, [3,1], dim=0)
        rgb = unorm(rgb)
        writer.add_image(name + "RGB", rgb, num_steps)
        writer.add_image(name + "D", d, num_steps)

    if C == 6:
        # Untested haha
        back_rgb, fore_rgb = torch.split(img, [3,3], dim=0)
        back_rgb = unorm(back_rgb)
        fore_rgb = unorm(fore_rgb)
        writer.add_image(name + "foreground RGB", fore_rgb, num_steps)
        writer.add_image(name + "background RGB", back_rgb, num_steps)


    if C == 8:
        back_rgb, back_d, fore_rgb, fore_d = np.array_split(img,  [3,4,7],axis=0)
        fore_rgb = unorm(fore_rgb)
        back_rgb = unorm(back_rgb)
        writer.add_image(name + "foreground RGB", fore_rgb, num_steps)
        writer.add_image(name + "foreground D", fore_d, num_steps)
        writer.add_image(name + "background RGB", back_rgb, num_steps)
        writer.add_image(name + "background D", back_d, num_steps)
        

def clear_depth_layers(imgs):
    # For now, it will 0 out this layer so that there is no useful information
    _, C, _, _ = imgs.shape

    if C == 4:
        imgs[:,3] *= 0

    if C == 8:
        imgs[:,3] *= 0
        imgs[:,7] *= 0
    
    return imgs

def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):

        # forward
        real_a_cpu, real_b_cpu, coords_cpu = batch[0], batch[1], batch[2]

        if not opt.include_depth:
            real_a_cpu = clear_depth_layers(real_a_cpu)
            real_b_cpu = clear_depth_layers(real_b_cpu)

        if iteration == 1:
            add_images(real_a_cpu, writer, "Input Image A", epoch)
            add_images(real_b_cpu, writer, "Input Image B", epoch)

        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        coords.data.resize_(coords_cpu.size()).copy_(coords_cpu)

        fake_b = G(real_a)
        if iteration == 1:
            add_images(fake_b, writer, "Fake Image", epoch)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        # (Update D network less frequent as G network to let G learn more)
        ###########################

        if iteration % opt.step_ratio == 0:
            optimizerD.zero_grad()
        
            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = D.forward(fake_ab.detach())

            loss_d_fake = criterionGAN(pred_fake, False)
            if iteration == 1:
                writer.add_scalar("loss_discriminator_fake", loss_d_fake, epoch) 


            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = D.forward(real_ab)

            #writer.add_scalar("discriminator_entropy", entropy(pred_real), epoch)   TODO

            loss_d_real = criterionGAN(pred_real, True)
            if iteration == 1:
                writer.add_scalar("loss_discriminator_real", loss_d_real, epoch) 

        
            # Combined loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            if iteration == 1:
                writer.add_scalar("loss_discriminator", loss_d, epoch) 

            loss_d.backward()
       
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = D.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True) * opt.gan_bonus
        if iteration == 1:
            writer.add_scalar("loss_generator_gan", loss_g_gan, epoch) 

         # Second, G(A) = B

        loss_g_l1 = criterionL1(fake_b, real_b) * opt.l1_bonus
        if iteration == 1:
            writer.add_scalar("loss_generator_l1", loss_g_l1, epoch) 
        loss_g_sl = criterionSLL(fake_b, real_b, coords) * opt.sl_bonus
        if iteration == 1:
            writer.add_scalar("loss_generator_sl", loss_g_sl, epoch) 
        
        loss_g = loss_g_gan + loss_g_l1 + loss_g_sl
        if iteration == 1:
            writer.add_scalar("loss_generator", loss_g, epoch) 

        
        loss_g.backward()

        optimizerG.step()
        optimizerG.step()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d.data[0], loss_g.data[0]))


def validate(epoch):

    avg_psnr = 0
    for iteration, batch in enumerate(testing_data_loader, 1):
        if not opt.include_depth:
            batch[0] = clear_depth_layers(batch[0])
            batch[1] = clear_depth_layers(batch[1])

        input, target = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()


        add_images(input, writer, "Real Image(Validation)", epoch)
        prediction = G(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        if iteration == 1:
            add_images(prediction, writer, "Fake Image (Validation)", epoch)
    writer.add_scalar("PSNR (Validation)", avg_psnr, epoch)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
    net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
    torch.save(G, net_g_model_out_path)
    torch.save(D, net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

# Load data



# Make the model
if opt.restore_G is None:
    G = build_generator(opt.input_channels, opt.output_channels, opt.num_gen_filters, opt.num_gen_ds, opt.num_gen_blocks, 
                    norm_fn=nn.BatchNorm2d, use_dropout=False, gpu_ids=[0])
else:
    G = torch.load(opt.restore_G)

if opt.restore_D is None:
    D = build_discriminator(opt.input_channels + opt.output_channels, opt.num_dis_filters, norm_fn=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[0])
else:
    D = torch.load(opt.restore_D)


# Set up loss
criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()
criterionSLL = SLLoss()

# Set up optimizer
optimizerG = optim.Adam(list(G.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(list(D.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

print_network(G)
print_network(D)

# Train
real_a = torch.FloatTensor(opt.batchSize, opt.input_channels, 256, 256)
real_b = torch.FloatTensor(opt.batchSize, opt.output_channels, 256, 256)
coords = torch.FloatTensor(opt.batchSize, 4)

if opt.cuda:
    D = D.cuda()
    G = G.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionSLL = criterionSLL.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()
    coords = coords.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)
coords = Variable(coords)


for epoch in range(1, opt.nEpochs + 1):
    train(epoch)

    validate(epoch)
    if epoch % 10 == 0:

        checkpoint(epoch)



