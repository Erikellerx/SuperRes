from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.data import DIV2K
from baseline.SRCNN import SRCNN
from metrics import PSNR, SSIM
from args import get_args
from utils import build_model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(tqdm(loader, desc="Training", leave=False)):
        lr, hr = data
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 10 == 0:
            tqdm.write(f"Epoch: {epoch} Step {i}/{len(loader)}: Loss {loss.item()}")
            
    return running_loss / len(loader)


def validation(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    psnr = 0
    ssim = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc="Validation", leave=False)):
            lr, hr = data
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            running_loss += loss.item()
            
            psnr += PSNR(sr, hr)
            ssim += SSIM(sr, hr)
                  
    tqdm.write(f"\nEpoch: {epoch} Validation Loss: {running_loss / len(loader)}, PSNR: {psnr / len(loader)}, SSIM: {ssim / len(loader)}\n")
    return running_loss / len(loader), psnr, ssim


if __name__ == "__main__":
    
    args = get_args()
    
    #print args in a readable format
    print("======> Arguments: <======")
    for arg in vars(args):
        print(f"{arg}:", getattr(args, arg))
    print()

    train_dataset = DIV2K(args.data_dir, train=True)
    valid_dataset = DIV2K(args.data_dir, train=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.test_batchsize, shuffle=True, pin_memory=True)
    
    model = build_model(args.model)
    model.load_state_dict(torch.load("pretrain/ref_srcnn_x4.pth"))
    model.to(args.device)
    
    criterion = nn.MSELoss()
    '''optimizer = optim.AdamW([{"params": model.features.parameters()},
                           {"params": model.map.parameters()},
                           {"params": model.reconstruction.parameters(), "lr": args.lr * 0.1}],
                          lr=args.lr)'''
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    best_psnr = 0
    best_ssim = 0
    
    for epoch in range(1, 100):
        #train(model, train_loader, optimizer, criterion, args.device, epoch)
        _, psnr, ssim = validation(model, valid_loader, criterion, args.device, epoch)
        
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), f"checkpoint/{args.model}_3channel_psnr{psnr:.2f}.pth")
            tqdm.write("Best Model saved")
        
        if ssim > best_ssim:
            best_ssim = ssim
            torch.save(model.state_dict(), f"checkpoint/{args.model}_3channel_ssim{ssim:.2f}.pth")
            tqdm.write("Best Model saved")
        
    
    
    