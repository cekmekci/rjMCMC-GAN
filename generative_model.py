"""
Adapted from the original content available on the Github repository:

https://github.com/ETZET/MCMC_GAN

Changes made in this version:

1) Dependency on the "wandb" library is removed.

"""

from dis import dis
import os.path

import torch
from torch import nn
from torch.autograd import Variable
from torch import autograd
from process_data import ToyDataset, MinMaxScaler
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib
from scipy.stats import wasserstein_distance as EMD

class WGAN_SIMPLE(nn.Module):
    """
    Generative Model Architecture

    Model Architecture cited from Scheiter, M., Valentine, A., Sambridge, M., 2022. Upscaling
    and downscaling Monte Carlo ensembles with generative models, Geophys. J. Int., ggac100.

    This model use gradient penalty to enforce 1-Lipschitz constraint instead of Weight Clipping in the original paper.
    Citation: Gulrajani, Ahmed & Arjovsky. Improved training of wasserstein gans. Adv. Neural Inf. Process. Syst.
    """

    def __init__(self, ndim, nhid=300, nlatent=100, device="cpu"):
        """
        :param ndim: Number of feature in input data
        :param nhid: Number of hidden units per layer
        :param device: device on which a torch.Tensor is or will be allocated
        :param gen: Generator that consist of four layers of dropout layers with linear output
        :param disc: Discriminator that consist of four layers of dropout layers with linear output
        """
        super().__init__()

        self.ndim = ndim
        self.nlatent = nlatent
        self.device = device
        self.scaler = None

        self.gen = nn.Sequential(
            nn.Linear(self.nlatent,nhid),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(nhid,nhid),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(nhid,nhid),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(nhid,ndim),
        )

        self.disc = nn.Sequential(
            nn.Linear(self.ndim,nhid),
            nn.LeakyReLU(0.1),
            nn.Linear(nhid,nhid),
            nn.LeakyReLU(0.1),
            nn.Linear(nhid,nhid),
            nn.LeakyReLU(0.1),
            nn.Linear(nhid,1),
        )

        self.gen.apply(init_weights)
        self.disc.apply(init_weights)

        self.gen.to(device)
        self.disc.to(device)

    def normalize(self,data):
        self.scaler = MinMaxScaler()
        self.scaler.fit(data)
        return self.scaler.transform(data)


    def optimize(self, data, output_path, batch_size=128, lr=1e-4,
                 beta1=0.5, lambda_term=10, epochs=200, kkd=1, kkg=1, device="cpu"):

        # normalizae data to (-1,1) range
        data = self.normalize(data)
        # construct dataset and dataloader for batch training
        map_dataset = ToyDataset(data)
        dataloader = DataLoader(map_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=1)

        optimizer_gen = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_disc = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(beta1, 0.999))

        for epoch in range(epochs):
            for i, data_slice in enumerate(dataloader):
                tic = time.time()
                # Update Discriminator
                for _ in range(kkd):
                    optimizer_disc.zero_grad()

                    real_data = data_slice.to(device).float()
                    b_size = real_data.size(0)
                    fake_data = self.gen(torch.randn(b_size, self.nlatent, device=device).float())

                    gradient_penalty = self.calculate_gradient_penalty(real_data, fake_data, lambda_term)
                    D_loss_real = torch.mean(self.disc(real_data))
                    D_loss_fake = torch.mean(self.disc(fake_data))
                    score_disc = -D_loss_real + D_loss_fake + gradient_penalty

                    score_disc.backward()
                    optimizer_disc.step()

                # Update Generator
                for _ in range(kkg):
                    optimizer_gen.zero_grad()

                    real_data = data_slice.to(device)
                    b_size = real_data.size(0)
                    fake_data = self.gen(torch.randn(b_size, self.nlatent, device=device))
                    score_gen = -torch.mean(self.disc(fake_data))
                    score_gen.backward()

                    optimizer_gen.step()

                toc = time.time()
                # logging
                if i % 20 == 0:
                    print(
                        '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t Wasserstein Distance: %.4f\t  Elapsed time per Iteration: %.4fs'
                        % (epoch, epochs, i, len(dataloader),
                           score_disc, score_gen, (D_loss_real - D_loss_fake), (toc - tic)))

            # model saving
            model_save_path = os.path.join(output_path,"model")
            if not os.path.exists(model_save_path):
                os.mkdir(model_save_path)
            if epoch % 10 == 0 or epoch == epochs - 1:
                avg_diff,std_diff,dist = eval(self,data)
                print("average difference: {:.4f}/pixel, std difference: {:.4f}/pixel, \
                    EMD distance: {:.4f}/pixel".format(avg_diff,std_diff,dist))
                torch.save({
                    'ndim': self.ndim,
                    'scaler':self.scaler,
                    'epoch': epoch,
                    'EMD': dist,
                    'model_state_dict': self.state_dict(),
                }, "{}/model_epoch{}_EMD{:6f}.pth".format(model_save_path, epoch,dist))
            if epoch == epochs - 1:
                avg_diff,std_diff,dist = eval(self,data)
                print("average difference: {:.4f}/pixel, std difference: {:.4f}/pixel, \
                    EMD distance: {:.4f}/pixel".format(avg_diff,std_diff,dist))
                torch.save({
                    'ndim': self.ndim,
                    'scaler':self.scaler,
                    'epoch': epoch,
                    'EMD': dist,
                    'model_state_dict': self.state_dict(),
                }, "{}/model_final.pth".format(model_save_path))

    def calculate_gradient_penalty(self, real_images, fake_images, lambda_term):
        batch_size = real_images.shape[0]
        eta = torch.FloatTensor(batch_size, 1).uniform_(0, 1)
        eta = eta.expand(batch_size, real_images.size(1)).to(self.device)

        interpolated = eta * real_images + ((1 - eta) * fake_images).to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.disc(interpolated.float())

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                                  create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
        return grad_penalty

    def load(self,checkpoint):
        self.load_state_dict(checkpoint["model_state_dict"])
        self.scaler = checkpoint["scaler"]

    def generate(self,num=500):
        fake_data = np.zeros((num,self.ndim))
        # if num is divisible by 100, generate by batch, else generate one by one
        if num % 100 == 0:
            for i in range(int(num/100)):
                left_idx = 100 * i
                right_idx = 100 * (i+1)
                fake_data[left_idx:right_idx,:] = \
                        self.gen(torch.randn(100, self.nlatent, device=self.device)).cpu().detach().numpy()
        else:
            for i in range(num):
                fake_data[i,:] =  self.gen(torch.randn(1, self.nlatent, device=self.device)).cpu().detach().numpy()
        # scaler the data back to original range
        fake_data_scaled = self.scaler.inverse_transform(fake_data)
        del fake_data
        return fake_data_scaled


def eval(model, data):
    dim = data.shape[1]

    # generate fake data using Generator
    fake_data = np.zeros((5000,dim))
    for i in range(50):
        left_idx = 100 * i
        right_idx = 100 * (i+1)
        with torch.no_grad():
            fake_batch = model.gen(torch.randn(100, model.nlatent, device=model.device)).cpu().detach().numpy()
        fake_data[left_idx:right_idx,:] = fake_batch
    # compare mean
    real_avg = np.mean(data,axis=0)
    fake_avg = np.mean(fake_data,axis=0)
    avg_diff_pixel = np.sum(np.absolute(real_avg-fake_avg))/dim
    # compare std
    real_std = np.std(data,axis=0)
    fake_std = np.std(fake_data,axis=0)
    std_diff_pixel = np.sum(np.absolute(real_std-fake_std))/dim
    # calculate EMD distance
    distance = np.zeros(dim)
    for i in range(dim):
        distance[i] = EMD(data[:,i],fake_data[:,i])
    emd_dist_pixel = np.sum(distance)/dim

    return avg_diff_pixel,std_diff_pixel,emd_dist_pixel


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
