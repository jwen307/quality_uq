import torch
import torchvision
import os
from PIL import Image
import sys
import time
import numpy as np

sys.path.append('../../')



# Dataset that gets the pregenerated posteriors, true image, and measurement
class PosteriorDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, transforms=None):

        # Get the directories for all the images
        self.base_dir = base_dir
        self.x_dir = os.path.join(base_dir, 'x')
        self.x_hat_dir = os.path.join(base_dir, 'x_hat')
        self.y_dir = os.path.join(base_dir, 'y')

        self.x_files = sorted(os.listdir(self.x_dir))
        #self.y_files = sorted(os.listdir(self.y_dir))
        #self.x_hat_folders = sorted(os.listdir(self.x_hat_dir))



        if transforms is None:
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((256, 256)),

            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        #a = time.time()
        # Get the true image and the measurement
        x_image = Image.open(os.path.join(self.x_dir, f'{idx}.png'))
        y_image = Image.open(os.path.join(self.y_dir, f'{idx}.png'))
        #print(f'Getting x and y took {time.time() - a} seconds')

        #b = time.time()

        # Get the posterior samples
        x_hat_folder_path = os.path.join(self.x_hat_dir, str(idx))
        x_hat_images = [Image.open(os.path.join(x_hat_folder_path, img_file)) for img_file in os.listdir(x_hat_folder_path)]
        #x_hat_images = np.stack(x_hat_images, axis=0)
        #print(f'Getting x_hat took {time.time() - b} seconds')

        #c = time.time()

        x_tensor = self.transforms(x_image)
        y_tensor = self.transforms(y_image)
        x_hat_tensors = torch.stack([self.transforms(img) for img in x_hat_images])
        #x_hat_tensors = self.transforms(torch.from_numpy(x_hat_images))

        #print(f'Transforming took {time.time() - c} seconds')

        return y_tensor, x_tensor, x_hat_tensors





