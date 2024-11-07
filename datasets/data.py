import torch
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms


class DIV2K(Dataset):
    def __init__(self, data_dir,train = True, scale_factor=4, visualize=False):
     
        self.visualize = visualize
        self.patch_size = 48
        self.scale_factor = scale_factor
        self.train = train
     
        # Get all paths of images inside `data_dir` into a list
        self.lr_transform = transforms.Compose([
            #grayscale images are converted to 3 channels
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
  
        self.hr_transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
  
  
        if train:
            pattern = os.path.join(data_dir + f"{os.sep}DIV2k_train_HR{os.sep}", "*.png")
            self.hr_file_paths = sorted(glob.glob(pattern, recursive=True))
            pattern = os.path.join(data_dir + f"{os.sep}DIV2k_train_LR_bicubic{os.sep}", f"**{os.sep}*.png")
            self.lr_file_paths = sorted(glob.glob(pattern, recursive=True))
        else:
            pattern = os.path.join(data_dir + f"{os.sep}DIV2k_valid_HR{os.sep}", "*.png")
            self.hr_file_paths = sorted(glob.glob(pattern, recursive=True))
            pattern = os.path.join(data_dir + f"{os.sep}DIV2k_valid_LR_bicubic{os.sep}", f"**{os.sep}*.png")
            self.lr_file_paths = sorted(glob.glob(pattern, recursive=True))
            
               # Repeat each file path 9 times to create consistent 9 patches per image
            self.hr_file_paths = [fp for fp in self.hr_file_paths for _ in range(9)]
            self.lr_file_paths = [fp for fp in self.lr_file_paths for _ in range(9)]


        # Check if number of HR and LR images are same
        assert len(self.hr_file_paths) == len(self.lr_file_paths), "Number of HR and LR images are not same"


    def __len__(self):
        return len(self.hr_file_paths)

    def __getitem__(self, index):
        # Read HR and LR images using PIL
        hr_image = Image.open(self.hr_file_paths[index])
        lr_image = Image.open(self.lr_file_paths[index])

        # Apply transformation to images
        hr_image = self.hr_transform(hr_image)
        lr_image = self.lr_transform(lr_image)
  
        if self.train:

            lr_random_x = np.random.randint(0, lr_image.shape[1] - self.patch_size)
            lr_random_y = np.random.randint(0, lr_image.shape[2] - self.patch_size)

            hr_random_x = lr_random_x * self.scale_factor
            hr_random_y = lr_random_y * self.scale_factor

            lr_image_patched = lr_image[:, lr_random_x:lr_random_x + self.patch_size, lr_random_y:lr_random_y + self.patch_size]
            hr_image_patched = hr_image[:, hr_random_x:hr_random_x + self.patch_size * self.scale_factor, hr_random_y:hr_random_y + self.patch_size * self.scale_factor]
            
        else:
            # Determine evenly spaced 3x3 patch coordinates based on the index
            grid_size = 3  # 3x3 grid
            lr_height_step = lr_image.shape[1] // grid_size  # Step size for LR patches height
            lr_width_step = lr_image.shape[2] // grid_size   # Step size for LR patches width

            hr_height_step = hr_image.shape[1] // grid_size  # Step size for HR patches height
            hr_width_step = hr_image.shape[2] // grid_size   # Step size for HR patches width

            # Determine patch position within the 3x3 grid
            patch_index = index % 9  # 9 patches per image
            patch_x = (patch_index % grid_size) * lr_width_step
            patch_y = (patch_index // grid_size) * lr_height_step

            hr_patch_x = (patch_index % grid_size) * hr_width_step
            hr_patch_y = (patch_index // grid_size) * hr_height_step

            # Extract patches using appropriate step sizes
            lr_image_patched = lr_image[:, patch_y:patch_y + self.patch_size, patch_x:patch_x + self.patch_size]
            hr_image_patched = hr_image[:, hr_patch_y:hr_patch_y + self.patch_size * self.scale_factor, hr_patch_x:hr_patch_x + self.patch_size * self.scale_factor]



        if self.visualize:
            return lr_image_patched, hr_image_patched, lr_image, hr_image
        return lr_image_patched, hr_image_patched
     
     
     
if __name__ == "__main__":
    
    data_dir = "F:\superRes\datasets\DIV2K"
 
    # Create dataset object
    train_dataset = DIV2K(data_dir, train=True, visualize=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    

    # Iterate over data loader
    for lr_image_patched, hr_image_patched, lr_image, hr_image in train_dataset:
        print(lr_image_patched.shape, hr_image_patched.shape, lr_image.shape, hr_image.shape)
        
        # Visualize images
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(lr_image_patched.permute(1, 2, 0))
        plt.title("LR Image Patched")	
        plt.subplot(1, 2, 2)	
        plt.imshow(hr_image_patched.permute(1, 2, 0))	
        plt.title("HR Image Patched")
  
    
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(lr_image.permute(1, 2, 0))
        plt.title("LR Image")	
        plt.subplot(1, 2, 2)
        plt.imshow(hr_image.permute(1, 2, 0))
        plt.title("HR Image")
        plt.show()
        break
    
    