from tqdm import tqdm
import torch

from metrics import PSNR, SSIM

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