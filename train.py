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
from utils import get_model

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
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.test_batchsize, shuffle=True)
    
    model = get_model(args.model)
    model.to(args.device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_psnr = 0
    
    for epoch in range(1, 100):
        train(model, train_loader, optimizer, criterion, args.device, epoch)
        _, psnr, ssim = validation(model, valid_loader, criterion, args.device, epoch)
        
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), "checkpoint/srcnn_3channel.pth")
            tqdm.write("Best Model saved")
        
    
    
    