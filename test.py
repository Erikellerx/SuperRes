import sys
import pickle
import os
import argparse

from datasets import DIV2K
from metrics import PSNR, SSIM
from train_utils import validation
from utils import *

from args import print_args
import warnings

if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) == 0:
        print("Usage: python test.py experiment_name [data dir]")
        exit(1)
    
    experiment_name = argv[0]
    if experiment_name == "Interpolate":
        args = argparse.Namespace(
            model=experiment_name,
            data_dir='./data/DIV2K',
            gray_scale=False,
        )
        model = build_model(args)
        if os.path.exists(f"./logs/{experiment_name}/best_model.pth"):
            ckpt = torch.load(f"./logs/{experiment_name}/best_model.pth")
        else:
            ckpt = {}
            if not os.path.exists(f"./logs/{experiment_name}"):
                os.makedirs(f"./logs/{experiment_name}")
    else:
        if not os.path.exists(f"./logs/{experiment_name}/args.pkl"):
            print(f"Experiment {experiment_name} arguments not recorded. Using default arguments...")
            args = argparse.Namespace(
                model=experiment_name,
                data_dir='./data/DIV2K',
                gray_scale=False,
            )
        else:
            args = pickle.load(open(f"./logs/{experiment_name}/args.pkl", "rb"))
        print_args(args)

        ckpt = torch.load(f"./logs/{experiment_name}/best_model.pth")
        model = build_model(args)
        model.load_state_dict(ckpt['model'])
        model.eval()

    if 'psnr' in ckpt.keys() and 'ssim' in ckpt.keys():
        print(f"PSNR: {ckpt['psnr']}, SSIM: {ckpt['ssim']}")
    else:
        print("No PSNR and SSIM found in checkpoint. Calculating...")
        if len(argv) == 1:
            warnings.warn(f"No data directory specified. Using experiment data directory: {args.data_dir}")
        else:
            args.data_dir = argv[1]

        valid_dataset = DIV2K(args.data_dir, train=False, gray_scale=args.gray_scale)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if 'epoch' not in ckpt.keys():
            ckpt['epoch'] = -1

        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
        loss, psnr, ssim = validation(model.to(device), valid_loader, nn.MSELoss(), device, ckpt['epoch'])
        
        ckpt['psnr'] = psnr
        ckpt['ssim'] = ssim
        torch.save(ckpt, f"./logs/{experiment_name}/best_model.pth")