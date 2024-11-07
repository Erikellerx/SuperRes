import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

def rgb_to_y_channel(img):
    # Assuming img is a 3-channel RGB image with values in [0, 1]
    r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
    # Y channel formula for converting RGB to YCbCr
    y = 0.299 * r + 0.587 * g + 0.114 * b
    y = y.unsqueeze(1)
    return y

def PSNR(pred, target):
    # Convert pred and target to Y channel
    #if the input is one channel, then we don't need to convert it to Y channel
    if pred.shape[1] == 1:
        return 10 * torch.log10(1 / F.mse_loss(pred, target))
    pred_y = rgb_to_y_channel(pred)
    target_y = rgb_to_y_channel(target)
    return 10 * torch.log10(1 / F.mse_loss(pred_y, target_y))

def SSIM(pred, target):
    # Convert pred and target to Y channel
    if pred.shape[1] == 1:
        return structural_similarity_index_measure(pred, target)
    pred_y = rgb_to_y_channel(pred)
    target_y = rgb_to_y_channel(target)
    return structural_similarity_index_measure(pred_y, target_y)
