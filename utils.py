
from baseline.SRCNN import SRCNN
from baseline.FSRCNN import FSRCNN
from baseline.VDSR import VDSR
from baseline.ESPCN import ESPCN, espcn_x4
import torch

def build_model(model_name, resume):
    if model_name == 'SRCNN':
        print("======>  Baseline Model: SRCNN  <======")
        model = SRCNN()
        if resume:
            model.load_state_dict(torch.load('checkpoint\SRCNN\SRCNN_3channel_psnr26.18.pth'))
        return model
    elif model_name == 'FSRCNN':
        print("======>  Baseline Model: FSRCNN  <======")
        model = FSRCNN()
        if resume:
            model.load_state_dict(torch.load('checkpoint\FSRCNN\FSRCNN_3channel_psnr25.34.pth'))
        return model
    elif model_name == 'VDSR':
        print("======>  Baseline Model: VDSR  <======")
        return VDSR()
    elif model_name == 'ESPCN':
        print("======>  Baseline Model: ESPCN  <======")
        return espcn_x4(in_channels=3, out_channels=3, channels=64)
    else:
        raise ValueError("Model not found")