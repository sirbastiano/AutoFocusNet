import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d
import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        max_val = 1
        min_val = 0
    else:
        max_val = val_range[1]
        min_val = val_range[0]
    L = max_val - min_val

    padd = 0
    if window is None:
        real_size = min(window_size, img1.shape[-1], img1.shape[-2])
        window = create_window(real_size, img1.shape[1]).to(img1.device)
        padd = window_size // 2

    mu1 = conv2d(img1, window, padding=padd, groups=img1.shape[1])
    mu2 = conv2d(img2, window, padding=padd, groups=img2.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padd, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padd, groups=img2.shape[1]) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=padd, groups=img1.shape[1]) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def gaussian(window_size, sigma):
    gauss = torch.exp(-((torch.arange(window_size).float() - window_size // 2) ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr