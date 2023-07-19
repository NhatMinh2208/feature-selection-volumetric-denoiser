import torch
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.nn.functional as F

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def Ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim(img1, img2):
    return Ssim(img1, img2)

def psnr(img1, img2):
    return 20 * torch.log10(torch.tensor(1, dtype=torch.float32, device=img1.get_device())) - 10 * torch.log10(F.mse_loss(img1, img2))

def spec_mask(x):
    size = 11 # px
    kernel = torch.ones((3, 1, size, size)) / (size**2)
    if x.is_cuda: kernel = kernel.cuda()
    loc_mean = torch.squeeze(F.conv2d(torch.unsqueeze(x, dim=0), kernel, padding=size//2, groups=3), dim=0)
    return torch.clamp(x - loc_mean, 0, 1)

class SSIMLoss(SSIM):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        Ls = 1 - super(SSIMLoss, self).forward(img1, img2)
        return Ls

class L1SSIMLoss(SSIM):
    def __init__(self):
        super(L1SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        Ls = 1 - super(L1SSIMLoss, self).forward(img1, img2)
        L1 = F.l1_loss(img1, img2)
        return 0.5 * Ls + 0.5 * L1

class L2SSIMLoss(SSIM):
    def __init__(self):
        super(L2SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        Ls = 1 - super(L2SSIMLoss, self).forward(img1, img2)
        L2 = F.mse_loss(img1, img2)
        return 0.5 * Ls + 0.5 * L2

class L1HFENLoss(torch.nn.Module):
    def __init__(self):
        super(L1HFENLoss, self).__init__()
        self.kernel = torch.Tensor([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]]).repeat((3, 1, 1, 1))

    def forward(self, img1, img2):
        # L1 loss
        L1 = F.l1_loss(img1, img2)
        # HFEN loss
        if img1.is_cuda: self.kernel = self.kernel.cuda()
        log1 = F.conv2d(img1, self.kernel, padding=2, groups=3)
        log2 = F.conv2d(img2, self.kernel, padding=2, groups=3)
        Lhf = F.l1_loss(log1, log2)
        return 0.8 * L1 + 0.2 * Lhf

class L1HFENSpecLoss(L1HFENLoss):
    def __init__(self):
        super(L1HFENSpecLoss, self).__init__()
        self.kernel_mean = torch.ones((3, 1, 11, 11)) / (11**2)

    def forward(self, img1, img2):
        # L1 loss
        L1 = F.l1_loss(img1, img2)
        # HFEN loss
        if img1.is_cuda: self.kernel = self.kernel.cuda()
        log1 = F.conv2d(img1, self.kernel, padding=2, groups=3)
        log2 = F.conv2d(img2, self.kernel, padding=2, groups=3)
        Lhf = F.l1_loss(log1, log2)
        # spec loss
        if img1.is_cuda: self.kernel_mean = self.kernel_mean.cuda()
        lm1 = F.conv2d(img1, self.kernel_mean, padding=11//2, groups=3)
        lm2 = F.conv2d(img2, self.kernel_mean, padding=11//2, groups=3)
        Ls = F.l1_loss(torch.clamp(img1 - lm1, 0, 1), torch.clamp(img2 - lm2, 0, 1))
        return 0.8 * L1 + 0.1 * Lhf + 0.1 * Ls

class L1SpecLoss(torch.nn.Module):
    def __init__(self):
        super(L1SpecLoss, self).__init__()
        self.kernel_mean = torch.ones((3, 1, 11, 11)) / (11**2)

    def forward(self, img1, img2):
        # L1 loss
        L1 = F.l1_loss(img1, img2)
        # spec loss
        if img1.is_cuda: self.kernel_mean = self.kernel_mean.cuda()
        lm1 = F.conv2d(img1, self.kernel_mean, padding=11//2, groups=3)
        lm2 = F.conv2d(img2, self.kernel_mean, padding=11//2, groups=3)
        Ls = F.l1_loss(torch.clamp(img1 - lm1, 0, 1), torch.clamp(img2 - lm2, 0, 1))
        return 0.8 * L1 + 0.2 * Ls

class L1HFENTemporalLoss(L1HFENLoss):
    # from https://research.nvidia.com/sites/default/files/publications/dnn_denoise_author.pdf
    def __init__(self):
        super(L1HFENTemporalLoss, self).__init__()

    def forward(self, img1, img2):
        assert img1.dim() > 4 and img2.dim() > 4, "Temporal loss needs 5D data!"
        # L1 loss
        L1 = F.l1_loss(img1[:, img1.size(1)//2], img2[:, img2.size(1)//2])
        # HFEN loss
        if img1.is_cuda: self.kernel = self.kernel.cuda()
        log1 = F.conv2d(img1[:, img1.size(1)//2], self.kernel, padding=2, groups=3)
        log2 = F.conv2d(img2[:, img2.size(1)//2], self.kernel, padding=2, groups=3)
        Lhf = F.l1_loss(log1, log2)
        # temporal loss
        Lt = 0
        for i in range(1, img1.size(1)):
            Lt += F.l1_loss(torch.abs(img1[:, i] - img1[:, i-1]), torch.abs(img2[:, i] - img2[:, i-1]))
        Lt /= max(1, img1.size(1) - 1)
        return 0.8 * L1 + 0.1 * Lhf + 0.1 * Lt
    
class SMAPELoss(torch.nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-7  # Small constant to avoid division by zero
        numerator = torch.abs(y_pred - y_true)
        denominator = torch.abs(y_pred) + torch.abs(y_true) + epsilon
        smape = 2.0 * torch.mean(numerator / denominator)
        return smape