from baseline import SRCNN
from baseline import FSRCNN

from argparse import Namespace
import torch.optim as optim
import torch.nn as nn

import warnings

def build_model(args: Namespace):
    model_name = args.model
    if model_name == 'SRCNN':
        print("======>  Baseline Model: SRCNN  <======")
        return SRCNN(gray_scale=args.gray_scale)
    elif model_name == 'FSRCNN':
        print("======>  Baseline Model: FSRCNN  <======")
        return FSRCNN()
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