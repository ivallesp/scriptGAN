import torch
import torch.optim as optim
import torch.functional as F
import torch.nn as nn
import torch.autograd as autograd
from src.torch_frankenstein.normalization import LayerNorm

import numpy as np


# Discriminador

def init_weights(m):
    n_conv = 0
    n_lin = 0
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        print("Linear layer initialized!")
    elif type(m)==nn.Conv1d:
        nn.init.xavier_uniform(m.weight)
        print("Conv1D layer initialized!")
    elif type(m)==nn.LSTMCell:
        nn.init.xavier_uniform(m.weight_ih)
        nn.init.xavier_uniform(m.weight_hh)
        print("LSTM cell initialized!")

class Discriminator(nn.Module):
    def __init__(self, channels_in, batch_size, cuda=True):
        super(Discriminator, self).__init__()
        dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.ln_in =  LayerNorm(channels_in)
        self.bn_d0 = nn.BatchNorm1d(1024)
        self.bn_d1 = nn.BatchNorm1d(512)
        self.bn_d2 = nn.BatchNorm1d(256)

        self.recurrent_hidden = (autograd.Variable(torch.zeros(1, batch_size, 1024).type(dtype)),
                                 autograd.Variable(torch.zeros(1, batch_size, 1024).type(dtype)))
        self.rnn = nn.LSTM(channels_in, 1024)
        self.d_1 = nn.Linear(1024, 512)
        self.d_2 = nn.Linear(512, 256)
        self.d_3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.ln_in.forward(x)
        lstm_out, _ = self.rnn(x.permute(2,0,1), self.recurrent_hidden)
        lstm_out = lstm_out.permute(1,2,0)[:,:,-1]
        o_1 = nn.LeakyReLU()(self.d_1(self.bn_d0(lstm_out)))
        o_2 = nn.LeakyReLU()(self.d_2(self.bn_d1(o_1)))
        o_3 = self.d_3(self.bn_d2(o_2))
        return o_3


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
            out = hx
            output.append(out)

        output = torch.stack(output, dim=2)

        output_stack = output.view(-1, output.size(1))
        o_1 = nn.LeakyReLU()(self.d_1(self.bn_d0(output_stack)))
        o_2 = nn.LeakyReLU()(self.d_2(self.bn_d1(o_1)))
        o_3 = self.d_3(self.bn_d2(o_2))
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

    gradient_penalty = ((gradients.contiguous().view(batch_size, -1).norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

def calc_gradient_penalty_slogan(real_data, fake_data, errD_real_vec, errD_fake_vec):
    clip_fn = lambda x: x.clamp(max=0)
    dist = ((real_data - fake_data) ** 2).sum(1).sum(1) ** 0.5
    lip_est = (errD_real_vec - errD_fake_vec).squeeze().abs() / (dist + 1e-8)
    lip_loss = (clip_fn(1.0 - lip_est) ** 2)
    return(lip_loss)

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
                      max_length=self.max_length).cuda()
        D = Discriminator(channels_in=self.n_outputs, batch_size=self.batch_size).cuda()
        D.apply(init_weights)
        G.apply(init_weights)
        dOptimizer = optim.Adam(D.parameters(), lr=1e-5)
        gOptimizer = optim.Adam(G.parameters(), lr=1e-5)
        return {"G": G, "D": D, "dOptimizer": dOptimizer, "gOptimizer": gOptimizer}

    def define_losses(self):
        def calculate_D_cost(G, D, real_data, z):
            fake_data = G.forward(z)
            D_real = D(real_data)
            D_fake = D(fake_data)
            lip_loss = calc_gradient_penalty_slogan(real_data, fake_data, D_real, D_fake)
            D_cost = D_real - D_fake + lip_loss * 10
            return D_cost.mean()

        def calculate_G_cost(G, D, z):
            fake_data = G.forward(z)
            G_cost = D(fake_data)
            return G_cost.mean()

        def calculate_W_approx(G, D, real_data, z):
            fake_data = G.forward(z)
            D_real = D(real_data)
            D_fake = D(fake_data)
            W_approx = D_real - D_fake
            return W_approx.mean()

        return {"D": calculate_D_cost, "G": calculate_G_cost, "W": calculate_W_approx}

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
        sw_w_approx = lambda sw, x, c: sw.add_scalar(self.name + "/Summaries/" + self.name + "/" + "Wasserstein", x, c)

        sw_gan_equilibrium = lambda sw, x, c: sw.add_scalar(
            self.name + "/Summaries/" + self.name + "/" + "GAN_Equilibrium", x, c)
        sw_accuracy_1 = lambda sw, x, c: sw.add_scalar(self.name + "/Summaries/" + self.name + "/" + "1_gram_accuracy",
                                                       x, c)
        sw_accuracy_2 = lambda sw, x, c: sw.add_scalar(self.name + "/Summaries/" + self.name + "/" + "2_gram_accuracy",
                                                       x, c)
        sw_accuracy_3 = lambda sw, x, c: sw.add_scalar(self.name + "/Summaries/" + self.name + "/" + "3_gram_accuracy",
                                                       x, c)

        def loss_summaries(sw, g_loss, d_loss, w_approx, c):
            sw_g_loss(sw, g_loss, c)
            sw_d_loss(sw, d_loss, c)
            sw_w_approx(sw, w_approx, c)
            sw_gan_equilibrium(sw, d_loss - g_loss, c)

        return {"loss_summaries": loss_summaries, "acc_1": sw_accuracy_1, "acc_2": sw_accuracy_2,
                "acc_3": sw_accuracy_3}


__architectures__ = {"GAN": GAN}
