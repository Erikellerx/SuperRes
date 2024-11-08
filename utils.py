import torch.optim as optim
import torch.nn as nn
import torch

from matplotlib import pyplot as plt
from matplotlib import patches

from baseline.SRCNN import SRCNN
from baseline.FSRCNN import FSRCNN
from baseline.VDSR import VDSR
from baseline.ESPCN import ESPCN, espcn_x4
from baseline.IDN import IDN
from baseline.EDSR import EDSR
from baseline import Interpolate

from datasets import DIV2K

from metrics import PSNR, SSIM

from argparse import Namespace
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
    


def visualize(ds: DIV2K, models: list[nn.Module], model_names: list[str], device: torch.device, max_num: int = 10):
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
        
        num_axes = 2 + len(models)
        fig, axes = plt.subplots(1, num_axes, figsize=(5 * num_axes, 10))

        patch_width = patch_height = patch_size * scale_factor
        axes[0].imshow(hr_image.permute(1, 2, 0).numpy())
        axes[0].add_patch(patches.Rectangle(hr_xy, patch_width, patch_height, linewidth=3, edgecolor='r', facecolor='none'))
        axes[0].set_title("HR")
        axes[0].axis('off')

        axes[1].imshow(hr_image_patched[0].permute(1, 2, 0).numpy())
        axes[1].set_title("HR Ground Truth")
        axes[1].axis('off')

        for j, model in enumerate(models):
            if isinstance(model, Interpolate):
                model_name = model.mode
            else:
                model_name = model_names[j]

            model = model.to(device)
            sr = model(lr_image_patched.to(device)).detach().cpu()
            psnr = PSNR(sr, hr_image_patched)
            ssim = SSIM(sr, hr_image_patched)

            sr = sr.squeeze(0).permute(1, 2, 0).numpy()
            axes[2 + j].imshow(sr)
            axes[2 + j].set_title(f"{model_name}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
            axes[2 + j].axis('off')
        
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