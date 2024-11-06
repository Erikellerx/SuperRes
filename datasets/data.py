import os
import torch  # Ensure torch is imported
from datasets import load_dataset, DownloadConfig
import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np




# Custom Dataset class to handle high and low resolution images
class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        lr_image = Image.open(item['lr'])  # Low-resolution image path
        hr_image = Image.open(item['hr'])  # High-resolution image path
        
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        
        return lr_image, hr_image


if __name__ == "__main__":
    
    data_dir = "F:\superRes\datasets\div2k"
    
    

    train_dataset = load_dataset("eugenesiow/Div2k", split="train", cache_dir = data_dir, trust_remote_code=True)
    val_dataset = load_dataset("eugenesiow/Div2k", split="validation", cache_dir = data_dir, trust_remote_code=True)
    
    print(train_dataset.cache_files)
    
    print("Dataset Cach Location:", train_dataset.cache_files[0]['filename'])
    
    
    transform = transforms.Compose([
        transforms.ToTensor()        
    ])
    
    div2k_train = DIV2KDataset(train_dataset, transform=transform)
    div2k_val = DIV2KDataset(val_dataset, transform=transform)
    
    print("Number of training samples:", len(div2k_train))
    print("Number of validation samples:", len(div2k_val))
    
    # Create DataLoader
    train_dataloader = DataLoader(div2k_train, batch_size=8, shuffle=True)
    


    for lr_images, hr_images in train_dataloader:
        print("Low-resolution batch shape:", lr_images.shape)
        print("High-resolution batch shape:", hr_images.shape)
        
        #visualize the images
        
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        for i in range(4):
            ax[0, i].imshow(np.transpose(lr_images[i].numpy(), (1, 2, 0)))
            ax[1, i].imshow(np.transpose(hr_images[i].numpy(), (1, 2, 0)))
        
        break
