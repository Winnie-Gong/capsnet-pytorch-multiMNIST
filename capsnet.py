import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16, iter_n=3):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

        self.iter_n = iter_n

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x) #[100, 1152, 10, 16, 1]

        b_ij = Variable(torch.zeros(batch_size, self.num_routes, self.num_capsules, 1, 1)) #[100, 1152, 10, 1, 1]
        if USE_CUDA:
            b_ij = b_ij.cuda()

        for iteration in range(self.iter_n):
            c_ij = F.softmax(b_ij, dim=1) #[100, 1152, 10, 1, 1]
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) #[100, 1, 10, 16, 1]
            v_j = self.squash(s_j) #[100, 1, 10, 16, 1]

            if iteration < self.iter_n - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1)) #[100, 1152, 10, 1, 1]
                b_ij = b_ij + a_ij
        return v_j.squeeze(1), u_hat[0, :, :, :, 0], v_j[0, 0, :, :, 0]

    def squash(self, input_tensor):
        squared_norm = torch.square(torch.norm(input_tensor))
        output_tensor = squared_norm/(1+squared_norm)/torch.sqrt(squared_norm)*input_tensor
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, input_width=28, input_height=28, input_channel=1, multi=False):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstruction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_height * self.input_channel),
            nn.Sigmoid()
        )
        self.multi = multi

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes, dim=0)

        if not self.multi:
            _, max_length_indices = classes.max(dim=1)
            max_length_indices = max_length_indices.squeeze()
            masked = Variable(torch.sparse.torch.eye(10))
            if USE_CUDA:
                masked = masked.cuda()
            masked = masked.index_select(dim=0, index=Variable(max_length_indices.data))
            t = (x * masked[:, :, None, None]).view(x.size(0), -1)
            reconstructions = self.reconstruction_layers(t)
        else:
            _, max_length_indices = torch.topk(classes, 2, dim=1)
            max_length_indices = max_length_indices.squeeze()
            masked = Variable(torch.sparse.torch.eye(10))
            if USE_CUDA:
                masked = masked.cuda()
            masked_cap0 = masked.index_select(dim=0, index=Variable(max_length_indices[:,0].data))
            t0 = (x * masked_cap0[:, :, None, None]).view(x.size(0), -1)
            masked_cap1 = masked.index_select(dim=0, index=Variable(max_length_indices[:,1].data))
            t1 = (x * masked_cap1[:, :, None, None]).view(x.size(0), -1)
            masked = torch.add(masked_cap0, masked_cap1)
            reconstructions = torch.add(self.reconstruction_layers(t0), self.reconstruction_layers(t1))
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked



class CapsNet(nn.Module):
    def __init__(self, config=None, multi=False, iter_n=3):
        super(CapsNet, self).__init__()
        if config:
            self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
            self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels,
                                                config.pc_kernel_size, config.pc_num_routes)
            self.digit_capsules = DigitCaps(config.dc_num_capsules, config.dc_num_routes, config.dc_in_channels,
                                            config.dc_out_channels, iter_n=iter_n)
            self.decoder = Decoder(config.input_width, config.input_height, config.cnn_in_channels, multi=multi)
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()
            self.decoder = Decoder()

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output, self.routing_in, self.routing_out = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005
