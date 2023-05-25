import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MNIST:

    @staticmethod
    def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor):
        _data = x.clone()
        _data = _data.view(-1, _data.numel() // _data.shape[0]) # flatten the image
        overlay = torch.zeros(_data.shape[0], 10)
        overlay[range(_data.shape[0]), y] = x.max()
        _data[:, :10] = overlay 
        return _data, y    
    
    @staticmethod
    def negative_data_generation(a: torch.Tensor, b: torch.Tensor, blurring_steps: int = 3):
        """
        Create negative data that has very long correlations but very similar short range correlations. 
        :a: an individual image from the dataset
        :b: an individual image from the dataset
        :blurring_steps: number of steps to apply a blurring filter on the image
        """

        mask = torch.rand((a.shape)) < 0.5
        mask = mask.int().float()
        filter = torch.tensor([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
        filter = filter.view(1, 1, 3, 3)
        mask = mask.view(1, 1, 28, 28)
        for step in range(blurring_steps):
            mask = torch.nn.functional.conv2d(mask, filter, padding="same")

        # threshold the image at 0.5
        mask = mask < 0.5
        mask = mask.int().view(28, 28)
        reversed_mask = torch.bitwise_not(mask.bool()).int().view(28, 28)
        neg_data = torch.mul(mask, a) + torch.mul(reversed_mask, b)
        return neg_data

    @staticmethod
    def predict(data_loader: torch.utils.data.DataLoader, model: nn.Module, device="cpu"):
        model.eval()
        predictions = []
        real = []
        for i, (x, y) in enumerate(data_loader):
            goodness = []
            for num in range(10):
                over_x = MNIST.overlay_y_on_x(x, num)
                over_x = over_x[0].to(device)
                good = model(over_x)
                goodness.append(good.detach().cpu().numpy())
            
            goodness = np.array(goodness).T
            predictions.extend(np.argmax(goodness,axis = 1))
            real.extend(list(y.detach().cpu().numpy()))
        return np.array(predictions), np.array(real)
        

class MergedDataset(Dataset):
    def __init__(self, original_dataset, steps, negative_data_generator = MNIST.negative_data_generation):
        self.original_dataset = original_dataset
        self.num_images = len(original_dataset)
        self.steps = steps
        self.negative_data_generator = negative_data_generator

    def __getitem__(self, index):
        # Randomly select two images from the original dataset
        index1 = random.randint(0, self.num_images - 1)
        index2 = random.randint(0, self.num_images - 1)
        img1, _ = self.original_dataset[index1]
        img2, _ = self.original_dataset[index2]
        return self.negative_data_generator(img1, img2, self.steps)

    def __len__(self):
        # Return the number of pairs we can make from the original dataset
        return self.num_images

class FashionMNIST:

    @staticmethod
    def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor):
        _data = x.clone()
        _data = _data.view(-1, _data.numel() // _data.shape[0]) # flatten the image
        overlay = torch.zeros(_data.shape[0], 10)
        overlay[range(_data.shape[0]), y] = x.max()
        _data[:, :10] = overlay 
        return _data, y    
    
    @staticmethod
    def negative_data_generation(a: torch.Tensor, b: torch.Tensor, blurring_steps: int = 3):
        """
        Create negative data that has very long correlations but very similar short range correlations. 
        :a: an individual image from the dataset
        :b: an individual image from the dataset
        :blurring_steps: number of steps to apply a blurring filter on the image
        """

        mask = torch.rand((a.shape)) < 0.5
        mask = mask.int().float()
        filter = torch.tensor([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
        filter = filter.view(1, 1, 3, 3)
        mask = mask.view(1, 1, 28, 28)
        for step in range(blurring_steps):
            mask = torch.nn.functional.conv2d(mask, filter, padding="same")

        # threshold the image at 0.5
        mask = mask < 0.5
        mask = mask.int().view(28, 28)
        reversed_mask = torch.bitwise_not(mask.bool()).int().view(28, 28)
        neg_data = torch.mul(mask, a) + torch.mul(reversed_mask, b)
        return neg_data

    @staticmethod
    def predict(data_loader: torch.utils.data.DataLoader, model: nn.Module, device="cpu"):
        model.eval()
        predictions = []
        real = []
        for i, (x, y) in enumerate(data_loader):
            goodness = []
            for num in range(10):
                over_x = FashionMNIST.overlay_y_on_x(x, num)
                over_x = over_x[0].to(device)
                good = model(over_x)
                goodness.append(good.detach().cpu().numpy())
            
            goodness = np.array(goodness).T
            predictions.extend(np.argmax(goodness,axis = 1))
            real.extend(list(y.detach().cpu().numpy()))
        return np.array(predictions), np.array(real)

class CIFAR10:

    @staticmethod
    def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor):
        _data = x.clone()
        _data = _data.view(-1, _data.numel() // _data.shape[0]) # flatten the image
        overlay = torch.zeros(_data.shape[0], 10)
        overlay[range(_data.shape[0]), y] = x.max()
        _data[:, :10] = overlay 
        return _data, y    

    @staticmethod
    def negative_data_generation(a: torch.Tensor, b: torch.Tensor, blurring_steps: int = 3):
        """
        Create negative data that has very long correlations but very similar short range correlations. 
        :a: an individual image from the dataset
        :b: an individual image from the dataset
        :blurring_steps: number of steps to apply a blurring filter on the image
        """
        mask = torch.rand_like(a) < 0.5
        mask = mask.float()
        filter = torch.tensor([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]).to(a.device)
        filter = filter.view(1, 1, 3, 3).repeat(3, 1, 1, 1)  # Repeat for 3 color channels

        for step in range(blurring_steps):
            mask = torch.nn.functional.conv2d(mask, filter, padding=1, groups=3)  # Apply filter independently for each channel

        # threshold the image at 0.5
        mask = mask < 0.5
        mask = mask.int()
        reversed_mask = torch.bitwise_not(mask.bool()).int()
        neg_data = torch.mul(mask, a) + torch.mul(reversed_mask, b)
        return neg_data

    @staticmethod
    def predict(data_loader: torch.utils.data.DataLoader, model: nn.Module, device="cpu"):
        model.eval()
        predictions = []
        real = []
        for i, (x, y) in enumerate(data_loader):
            goodness = []
            for num in range(10):
                over_x = CIFAR10.overlay_y_on_x(x, num)
                over_x = over_x[0].to(device)
                good = model(over_x)
                goodness.append(good.detach().cpu().numpy())

            goodness = np.array(goodness).T
            predictions.extend(np.argmax(goodness, axis=1))
            real.extend(list(y.detach().cpu().numpy()))
        return np.array(predictions), np.array(real)
