# external imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------
# autoencoder modules

class Encoder(nn.Module):
    def __init__(self, f_in, f_out):
        super(Encoder, self).__init__()
        self.add_module('conv', nn.Conv2d(f_in, f_out, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        skip = x
        x = F.leaky_relu(self.conv(x), negative_slope=0.1)
        return F.max_pool2d(x, kernel_size=2), skip

class Decoder(nn.Module):
    def __init__(self, f_in, f_out):
        super(Decoder, self).__init__()
        self.add_module('conv1', nn.ConvTranspose2d(f_in, f_out, kernel_size=3, stride=1, padding=1))
        self.add_module('conv2', nn.ConvTranspose2d(f_out, f_out, kernel_size=3, stride=1, padding=1))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.size()[-2:])
        x = torch.cat((x, skip), dim=-3)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_channels, output_channels=3, stages=5, hidden=48):
        super(Autoencoder, self).__init__()
        stages = max(2, stages) # ensure min 2 stages
        # input layer
        self.add_module("input", nn.Conv2d(input_channels, hidden, kernel_size=3, stride=1, padding=1))
        # encoder modules
        self.encoders = nn.ModuleList()
        for i in range(stages):
            self.encoders.append(Encoder(hidden, hidden))
        # latent space
        self.add_module("latent", nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1))
        # decoder modules
        self.decoders = nn.ModuleList()
        for i in range(stages):
            if i == 0: # first decoder: hidden + skip_connections feature maps
                self.decoders.append(Decoder(2*hidden, 2*hidden))
            else: # intermediate decoder: 2*hidden + skip_connections feature maps
                self.decoders.append(Decoder(3*hidden, 2*hidden))
        # output layer
        self.add_module('dropout', nn.Dropout2d(0.1))
        self.add_module("output", nn.ConvTranspose2d(2*hidden, output_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.input(x * 2 - 1) # zero centered input
        # encoder stages
        skip_connections = []
        for i, encoder in enumerate(self.encoders):
            x, skip = encoder(x)
            skip_connections.append(skip)
        # latent space
        x = self.latent(x)
        # decoder stages
        skip_connections.reverse()
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        x = self.dropout(x)
        return self.output(x) * 0.5 + 0.5 # output in [0, 1]

class AutoencoderDualF24(nn.Module):
    def __init__(self):
        super(AutoencoderDualF24, self).__init__()
        #self.add_module("aec_direct", Autoencoder(15, 3, stages=5, hidden=48))
        #self.add_module("aec_indirect", Autoencoder(15, 3, stages=5, hidden=48))
        self.add_module("aec_direct", Autoencoder(26, 3, stages=5, hidden=48))
        self.add_module("aec_indirect", Autoencoder(28, 3, stages=5, hidden=48))
        self.add_module("output", nn.ConvTranspose2d(6, 3, kernel_size=3, stride=1, padding=1))

#     def forward(self, x):
#         direct = self.aec_direct(x[..., 0:15, :, :])
#         indirect = self.aec_indirect(torch.cat((x[..., 0:3, :, :], direct, x[..., 15:24, :, :]), dim=-3))
#         x = self.output(torch.cat((direct * 2 - 1, indirect * 2 - 1), dim=-3)) * 0.5 + 0.5
#         return x
    def forward(self, x_noisy, x_ss, x_ms):
        direct = self.aec_direct(torch.cat((x_noisy, x_ss), dim = -3))
        indirect = self.aec_indirect(torch.cat((x_noisy, direct, x_ms), dim=-3))
        x = self.output(torch.cat((direct * 2 - 1, indirect * 2 - 1), dim=-3)) * 0.5 + 0.5
        return x

class AutoencoderDualF24Big(nn.Module):
    def __init__(self):
        super(AutoencoderDualF24Big, self).__init__()
        self.add_module("aec_direct", Autoencoder(15, 12, stages=5, hidden=48))
        self.add_module("aec_indirect", Autoencoder(24, 12, stages=5, hidden=48))
        self.add_module("output", nn.ConvTranspose2d(24, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        direct = self.aec_direct(x[..., 0:15, :, :])
        indirect = self.aec_indirect(torch.cat((x[..., 0:3, :, :], direct, x[..., 15:24, :, :]), dim=-3))
        x = self.output(torch.cat((direct * 2 - 1, indirect * 2 - 1), dim=-3)) * 0.5 + 0.5
        return x
