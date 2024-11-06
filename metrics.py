import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

def PSNR(pred, target):
    return 10 * torch.log10(1 / F.mse_loss(pred, target))

def SSIM(pred, target):
    return structural_similarity_index_measure(pred, target)

