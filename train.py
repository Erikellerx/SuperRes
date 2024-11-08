from tqdm import tqdm
import os
import torch
import pickle
import tensorboardX

from datasets.data import DIV2K
from args import get_args
from utils import *
from train_utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    
    args = get_args()

    logdir = os.path.join(args.log_dir, args.experiment_name)
    args_path = os.path.join(logdir, "args.pkl")
    
    best_model_path = os.path.join(logdir, "best_model.pth")
    latest_model_path = os.path.join(logdir, "latest_model.pth")

    if os.path.exists(logdir):
        if args.resume:
            print(f"Resuming experiment {args.experiment_name}")
            with open(args_path, "rb") as f:
                args = pickle.load(f)
            args.resume = True
        else:
            s = input(f"Experiment {args.experiment_name} already exists. Do you want to continue? (y/n) ")
            if s != "y":
                exit(0)
    else:
        os.makedirs(logdir)
    
    with open(args_path, "wb") as f:
        pickle.dump(args, f)
    
    #print args in a readable format
    print("======> Arguments: <======")
    for arg in vars(args):
        print(f"{arg}:", getattr(args, arg))
    print()

    train_dataset = DIV2K(args.data_dir, train=True, gray_scale=args.gray_scale)
    valid_dataset = DIV2K(args.data_dir, train=False, gray_scale=args.gray_scale)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.test_batch)
    
    model = build_model(args)

    model.to(args.device)
    
    criterion = build_criterion(args)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    if args.resume:
        ckpt = torch.load(latest_model_path)
        model.load_state_dict(ckpt['model'])
        tqdm.write("Latest Model loaded")
        scheduler.load_state_dict(ckpt['scheduler'])
        optimizer.load_state_dict(ckpt['optimizer'])
        tqdm.write("Optimizer and Scheduler loaded")

        starting_epoch = ckpt['epoch']
    else:
        starting_epoch = 0

    writer = tensorboardX.SummaryWriter(logdir)

    best_psnr = 0
    for epoch in range(starting_epoch, args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, args.device, epoch + 1)
        valid_loss, psnr, ssim = validation(model, valid_loader, criterion, args.device, epoch + 1)
        
        saved_dict = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'psnr': psnr / len(valid_loader),
            'ssim': ssim / len(valid_loader)
        }
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(saved_dict, best_model_path)

        writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch + 1)
        writer.add_scalar("Loss/val", valid_loss / len(valid_loader), epoch + 1)
        writer.add_scalar("PSNR/val", psnr / len(valid_loader), epoch + 1)
        writer.add_scalar("SSIM/val", ssim / len(valid_loader), epoch + 1)
        writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch + 1)

        torch.save(saved_dict, latest_model_path)
        
        scheduler.step()