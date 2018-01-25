## import torch
import torch.optim as optim
import torch.functional as F
import torch.nn as nn
import torch.autograd as autograd

import numpy as np


# Discriminador

class Discriminator(nn.Module):
    def __init__(self, channels_in):
        super(Discriminator, self).__init__()
        self.c11 = nn.Conv1d(in_channels=channels_in, out_channels=64, kernel_size=9, padding=4)
        self.c12 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9, padding=4)
        self.p1 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.c21 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, padding=4)
        self.c22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, padding=4)
        self.p2 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.c31 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9, padding=4)
        self.c32 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=9, padding=4)
        self.p3 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.c41 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=9, padding=4)
        self.c42 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=9, padding=4)
        self.p4 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.c5 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=4)

    def forward(self, x):
        x = nn.LeakyReLU()(self.c11(x))
        x = nn.LeakyReLU()(self.c12(x))
        x = self.p1(x)

        x = nn.LeakyReLU()(self.c21(x))
        x = nn.LeakyReLU()(self.c22(x))
        x = self.p2(x)

        x = nn.LeakyReLU()(self.c31(x))
        x = nn.LeakyReLU()(self.c32(x))
        x = self.p3(x)

        x = nn.LeakyReLU()(self.c41(x))
        x = nn.LeakyReLU()(self.c42(x))
        x = self.p4(x)

        x = self.c5(x)
        return x


# Generator
class Generator(nn.Module):
    def __init__(self, noise_depth, batch_size, max_length, n_outputs, cuda=True):
        super(Generator, self).__init__()
        dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.bn_d_proj_zh = nn.BatchNorm1d(1024)
        self.bn_d_proj_zc = nn.BatchNorm1d(1024)
        self.bn_d0 = nn.BatchNorm1d(1024)
        self.bn_d1 = nn.BatchNorm1d(512)
        self.bn_d2 = nn.BatchNorm1d(256)

        self.max_length = max_length
        self.d_proj_zh = nn.Linear(noise_depth, 1024)
        self.d_proj_zc = nn.Linear(noise_depth, 1024)

        self.recurrent_cell = nn.LSTMCell(1024, 1024)
        self.d_1 = nn.Linear(1024, 512)
        self.d_2 = nn.Linear(512, 256)
        self.d_3 = nn.Linear(256, n_outputs)
        self.go = autograd.Variable(
            torch.from_numpy(np.array([1] + [0] * (1024 - 1))).type(dtype).view([1, -1]).repeat(batch_size, 1))

    def forward(self, z):
        zh = self.bn_d_proj_zh(self.d_proj_zh(z))
        zc = self.bn_d_proj_zc(self.d_proj_zc(z))
        hx, cx = zh, zc
        out = self.go

        output = []
        for i in range(self.max_length):
            hx, cx = self.recurrent_cell(out, (hx, cx))
            out = torch.tanh(hx)
            output.append(out)

        output = torch.stack(output, dim=2)

        output_stack = output.view(-1, output.size(1))
        o_1 = nn.LeakyReLU()(self.d_1(self.bn_d0(output_stack)))
        o_2 = nn.LeakyReLU()(self.d_2(self.bn_d1(o_1)))
        o_3 = nn.LeakyReLU()(self.d_3(self.bn_d2(o_2)))
        output_unstack = o_3.view(output.size(0), -1, output.size(2))
        o = nn.Softmax(dim=1)(output_unstack)
        return o


def calc_wasserstein_gradient_penalty(D, real_data, fake_data, cuda=True):
    assert real_data.size(0) == fake_data.size(0)
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = D.forward(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.view(batch_size, -1).norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class GAN:
    def __init__(self, noise_depth=100, batch_size=512, n_outputs=104, max_length=64, name="GAN"):
        self.noise_depth = noise_depth
        self.batch_size = batch_size
        self.n_outputs = n_outputs
        self.max_length = max_length
        self.name = name
        self.define_computation_graph()

        # Aliases
        self.op = self.optimizers
        self.summ = self.summaries

    def define_computation_graph(self):
        # Reset graph
        self.core_model = NameSpacer(**self.define_core_model())
        self.losses = NameSpacer(**self.define_losses())
        self.optimizers = NameSpacer(**self.define_optimizers())
        self.summaries = NameSpacer(**self.define_summaries())

    def define_core_model(self):
        G = Generator(noise_depth=self.noise_depth, batch_size=self.batch_size, n_outputs=self.n_outputs,
                      max_length=max_length).cuda()
        D = Discriminator(channels_in=self.n_outputs).cuda()
        dOptimizer = optim.Adam(D.parameters(), lr=15e-4)
        gOptimizer = optim.Adam(G.parameters(), lr=1e-4)
        return {"G": G, "D": D, "dOptimizer": dOptimizer, "gOptimizer": gOptimizer}

    def define_losses(self):
        def calculate_D_cost(G, D, real_data, z):
            fake_data = G.forward(z)
            gradient_penalty = calc_wasserstein_gradient_penalty(D, real_data.data, fake_data.data)
            D_real = D(real_data)
            D_fake = D(fake_data)
            D_cost = (D_fake - D_real + gradient_penalty * 10)
            return D_cost.mean()

        def calculate_G_cost(G, D, z):
            fake_data = G.forward(z)
            G_cost = -D(fake_data)
            return G_cost.mean()

        return {"D": calculate_D_cost, "G": calculate_G_cost}

    def define_optimizers(self):
        def optimize_D(real_data, z):
            cost = self.losses.D(self.core_model.G, self.core_model.D, real_data, z)
            self.core_model.dOptimizer.zero_grad()
            cost.backward()
            self.core_model.dOptimizer.step()
            return cost

        def optimize_G(z):
            cost = self.losses.G(self.core_model.G, self.core_model.D, z)
            self.core_model.gOptimizer.zero_grad()
            cost.backward()
            self.core_model.gOptimizer.step()
            return cost

        return {"G": optimize_G, "D": optimize_D}

    def define_summaries(self):
        sw_g_loss = lambda sw, x, c: sw.add_scalar(self.name + "/Summaries/" + self.name + "/" + "G_loss", x, c)
        sw_d_loss = lambda sw, x, c: sw.add_scalar(self.name + "/Summaries/" + self.name + "/" + "D_loss", x, c)
        sw_gan_equilibrium = lambda sw, x, c: sw.add_scalar(
            self.name + "/Summaries/" + self.name + "/" + "GAN_Equilibrium", x, c)
        sw_accuracy_1 = lambda sw, x, c: sw.add_scalar(self.name + "/Summaries/" + self.name + "/" + "1_gram_accuracy",
                                                       x, c)
        sw_accuracy_2 = lambda sw, x, c: sw.add_scalar(self.name + "/Summaries/" + self.name + "/" + "2_gram_accuracy",
                                                       x, c)
        sw_accuracy_3 = lambda sw, x, c: sw.add_scalar(self.name + "/Summaries/" + self.name + "/" + "3_gram_accuracy",
                                                       x, c)

        def loss_summaries(sw, g_loss, d_loss, c):
            sw_g_loss(sw, g_loss, c)
            sw_d_loss(sw, d_loss, c)
            sw_gan_equilibrium(sw, d_loss - g_loss, c)

        return {"loss_summaries": loss_summaries, "acc_1": sw_accuracy_1, "acc_2": sw_accuracy_2,
                "acc_3": sw_accuracy_3}


__architectures__ = {"GAN": GAN}
