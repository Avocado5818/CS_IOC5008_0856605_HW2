"""
@author: LU
"""

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

# custom weights initialization called on netG and netD
def weights_init(m_name):
    """ weights initialize """
    classname = m_name.__class__.__name__
    if classname.find('Conv') != -1:
        m_name.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m_name.weight.data.normal_(1.0, 0.02)
        m_name.bias.data.fill_(0)

# Generator model
class Generator(nn.Module):
    """ Generator model """
    def __init__(self, ngpu, opt):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(opt.nz, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, latent_input):
        output = self.main(latent_input)
        return output

# Discriminator model
class Discriminator(nn.Module):
    """ Discriminator model """
    def __init__(self, ngpu, opt):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image_input):
        output = self.main(image_input)
        return output.view(-1, 1).squeeze(1)

# Train module
def train():
    """ Train """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64,
                        help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='output_images',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    data_root = "../dataset/"
    dataset = datasets.ImageFolder(root=data_root,
                                   transform=transforms.Compose([
                                       transforms.CenterCrop((120, 80)),
                                       transforms.Resize((opt.imageSize, opt.imageSize)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    n_gpu = int(opt.ngpu)
    n_z = int(opt.nz)

    net_g = Generator(n_gpu, opt).to(device)
    net_g.apply(weights_init)
    if opt.netG != '':
        net_g.load_state_dict(torch.load(opt.netG))
    print(net_g)

    net_d = Discriminator(n_gpu, opt).to(device)
    net_d.apply(weights_init)
    if opt.netD != '':
        net_d.load_state_dict(torch.load(opt.netD))
    print(net_d)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(opt.batchSize, n_z, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    plot_1 = []
    plot_2 = []
    plot_3 = []
    plot_4 = []
    plot_5 = []

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            net_d.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = net_d(real_cpu)
            errd_real = criterion(output, label)
            errd_real.backward()
            d_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, n_z, 1, 1, device=device)
            fake = net_g(noise)
            label.fill_(fake_label)
            output = net_d(fake.detach())
            errd_fake = criterion(output, label)
            errd_fake.backward()
            d_g_z1 = output.mean().item()
            errd = errd_real + errd_fake
            optimizer_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            net_g.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = net_d(fake)
            errg = criterion(output, label)
            errg.backward()
            d_g_z2 = output.mean().item()
            optimizer_g.step()


            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errd.item(), errg.item(), d_x, d_g_z1, d_g_z2))
            if i % 100 == 0:
                plot_1.append(errg.item())
                plot_2.append(errg.item())
                plot_3.append(d_x)
                plot_4.append(d_g_z1)
                plot_5.append(d_g_z2)
                vutils.save_image(real_cpu, '%s/real_samples.png' % opt.outf, normalize=True)
                fake = net_g(fixed_noise)
                vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png'
                                  % (opt.outf, epoch), normalize=True)

        # do checkpointing
        torch.save(net_g.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(net_d.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

    np.save('errD', plot_1)
    np.save('errG', plot_2)
    np.save('D_x', plot_3)
    np.save('D_G_z1', plot_4)
    np.save('D_G_z2', plot_5)

if __name__ == '__main__':
    train()
