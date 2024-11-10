import torch.optim as optim
import torch.nn as nn
import torch

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import gridspec

from baseline import *

from datasets import DIV2K

from metrics import PSNR, SSIM

from argparse import Namespace
import math
import warnings

def build_model(args):
    model_name = args.model
    if model_name == 'SRCNN':
        print("======>  Baseline Model: SRCNN  <======")
        model = SRCNN()
        return model
    elif model_name == 'FSRCNN':
        print("======>  Baseline Model: FSRCNN  <======")
        model = FSRCNN()
        return model
    elif model_name == 'VDSR':
        print("======>  Baseline Model: VDSR  <======")
        return VDSR()
    elif model_name == 'ESPCN':
        print("======>  Baseline Model: ESPCN  <======")
        return espcn_x4(in_channels=3, out_channels=3, channels=64)
    elif model_name == 'IDN':
        print("======>  Baseline Model: IDN  <======")
        return IDN()
    elif model_name == 'EDSR':
        print("======>  Baseline Model: EDSR  <======")
        return EDSR(filters=256, n_resblock=32, res_scaling=0.1, scale=4)
    elif model_name == 'Interpolate':
        print("======>  Baseline Model: Interpolate  <======")
        return Interpolate()
    else:
        raise ValueError("Model not found")
    
def build_criterion(args: Namespace):
    if args.criterion == 'l1':
        return nn.L1Loss()
    elif args.criterion == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError("Criterion not found")
    
def build_optimizer(model: nn.Module, args: Namespace):
    argv = args.optimizer.split('+')

    if argv[0] == 'sgd':
        if args.model == 'SRCNN':
            return optim.SGD([
                {'params': model.features.parameters(), 'lr': float(argv[1])},
                {'params': model.map.parameters(), 'lr': float(argv[1])},
                {'params': model.reconstruction.parameters(), 'lr': float(argv[1]) * 0.1}
            ], momentum=float(argv[2]), weight_decay=float(argv[3]), nesterov=_str2bool(argv[4]))
        else:
            return optim.SGD(model.parameters(), lr=float(argv[1]), momentum=float(argv[2]), weight_decay=float(argv[3]), nesterov=_str2bool(argv[4]))
    elif argv[0] == 'adam':
        if args.model == 'SRCNN':
            return optim.Adam([
                {'params': model.features.parameters(), 'lr': float(argv[1])},
                {'params': model.map.parameters(), 'lr': float(argv[1])},
                {'params': model.reconstruction.parameters(), 'lr': float(argv[1]) * 0.1}
            ], eps=float(argv[2]), weight_decay=float(argv[3]))
        else:
            return optim.Adam(model.parameters(), lr=float(argv[1]), eps=float(argv[2]), weight_decay=float(argv[3]))
    else:
        print(
            """
optimizer (default: adam+1e-4+1e-8+0)
Two options available: sgd for adam.
For SGD: format is sgd+[lr]+[momentum]+[weight_decay]+[nesterov (True|False)]
For Adam: format is adam+[lr]+[eps]+[weight_decay]
            """
        )
        raise ValueError("Optimizer not found")
    
def build_scheduler(optimizer, args: Namespace):
    argv = args.scheduler.split('+')
    
    if argv[0] == 'none':
        return _NoneScheduler(optimizer)
    elif argv[0] == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=int(argv[1]), gamma=float(argv[2]))
    elif argv[0] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(argv[1]), eta_min=float(argv[2]))
    else:
        print(
            """
scheduler (default: none)
Two options available: step, cosine
For step: format is step+[step_size]+[gamma]
For cosine: format is cosine+[t_max]+[eta_min]            
            """
        )
        raise ValueError("Scheduler not found")
    


def visualize(ds: DIV2K, models: list[nn.Module], model_names: list[str], device: torch.device, max_num: int = 10, save_path='images'):
    if not ds.visualize:
        warnings.warn("Dataset is not set to visualize mode, skipping visualization.", UserWarning)
        return
    
    patch_size = ds.patch_size
    scale_factor = ds.scale_factor

    for i, (lr_image_patched, hr_image_patched, lr_image, hr_image, lr_xy, hr_xy) in enumerate(ds):

        if i == max_num:
            break

        lr_image_patched = lr_image_patched.unsqueeze(0)
        hr_image_patched = hr_image_patched.unsqueeze(0)

        patch_width = patch_height = patch_size * scale_factor

        # two columns for HR whole image, others are HR GT plus model reconstructions
        # in total two rows
        ncols = 2 + math.ceil((len(models) + 1) / 2)
        gs = gridspec.GridSpec(2, ncols)
        
        fig = plt.figure(figsize=(ncols * 5, 10))
        axes = []

        axes.append(fig.add_subplot(gs[0:2, 0:2]))
        axes[0].imshow(hr_image.permute(1, 2, 0).numpy())
        axes[0].add_patch(patches.Rectangle(hr_xy, patch_width, patch_height, linewidth=3, edgecolor='r', facecolor='none'))
        axes[0].set_title("HR", fontsize=15)
        axes[0].axis('off')

        axes.append(fig.add_subplot(gs[0, 2]))
        axes[1].imshow(hr_image_patched[0].permute(1, 2, 0).numpy())
        axes[1].set_title("HR Ground Truth", fontsize=15)
        axes[1].axis('off')

        best_psnr, best_ssim = (0., -1), (0., -1)
        psnrs, ssims = [], []
        for j, model in enumerate(models):
            row_idx = math.floor((j + 1) / (ncols - 2))
            col_idx = (j + 1) % (ncols - 2) + 2

            if isinstance(model, Interpolate):
                model_names[j] = model.mode

            model = model.to(device)
            sr = model(lr_image_patched.to(device)).detach().cpu()
            psnr = PSNR(sr, hr_image_patched)
            ssim = SSIM(sr, hr_image_patched)

            psnrs.append(psnr.item())
            ssims.append(ssim.item())

            if psnr > best_psnr[0]:
                best_psnr = (psnr, j)
            if ssim > best_ssim[0]:
                best_ssim = (ssim, j)

            sr = sr.squeeze(0).permute(1, 2, 0).numpy()

            axes.append(fig.add_subplot(gs[row_idx, col_idx]))
            axes[2 + j].imshow(sr)
            axes[2 + j].axis('off')
        
        for j in range(len(models)):
            title_lst = [model_names[j]]
            if best_psnr[1] == j:
                title_lst.append(rf"$\bf{{PSNR: {psnrs[j]:.2f}}}$")
            else:
                title_lst.append(f"PSNR: {psnrs[j]:.2f}")

            if best_ssim[1] == j:
                title_lst.append(rf"$\bf{{SSIM: {ssims[j]:.2f}}}$")
            else:
                title_lst.append(f"SSIM: {ssims[j]:.2f}")

            axes[2 + j].set_title(", ".join(title_lst), fontsize=15)
        
        # plt.tight_layout()
        plt.savefig(f"{save_path}/baselines_{i}.png", bbox_inches='tight')
        plt.show()




#################################
# Helper functions / classes

def _str2bool(s: str):
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError("Value not found")
    
class _NoneScheduler(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose="deprecated"):
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [group['lr'] for group in self.optimizer.param_groups]