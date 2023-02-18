import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# pip install opensimplex
import opensimplex
from torch.utils.data import DataLoader, Dataset

class CIFAR10:

    @staticmethod
    def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor):

        # torch.view acts as a pointer 
        _data = x.clone()
        # pad the image with zeros for the label
        _data = _data.view(-1, _data.numel() // _data.shape[0]) # flatten the image
        overlay = torch.zeros(_data.shape[0], 10)
        overlay[range(_data.shape[0]), y] = x.max()
        _data[:, :10] = overlay 
        return _data, y

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
            predictions.extend(np.argmax(goodness,axis = 1))
            real.extend(list(y.detach().cpu().numpy()))
        return np.array(predictions), np.array(real)

    @staticmethod
    def predict_resnet(data_loader: torch.utils.data.DataLoader, model: nn.Module, resnet, device="cpu"):
        model.eval()
        predictions = []
        real = []
        for i, (x, y) in enumerate(data_loader):
            goodness = []
            for num in range(10):
                over_x = CIFAR10.overlay_y_on_x(x, num)
                over_x = over_x[0].reshape((-1, 3, 32, 32))
                over_x = resnet(over_x.to(device))
                over_x = over_x.to(device)
                good = model(over_x)
                goodness.append(good.detach().cpu().numpy())
            goodness = np.array(goodness).T
            predictions.extend(np.argmax(goodness,axis = 1))
            real.extend(list(y.detach().cpu().numpy()))
        return np.array(predictions), np.array(real)


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
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_images = len(original_dataset)
        a = self.original_dataset[0][0][0]
        self.mask = torch.FloatTensor(np.array([[opensimplex.noise2(i,j) for i in range(a.shape[0])] for j in range(a.shape[1])])>0).reshape(-1)

    def merge_two_images(self, a: torch.Tensor, b: torch.Tensor):
        _a = a.clone()
        _a = _a.reshape(-1)
        _b = b.clone()
        _b = _b.reshape(-1)
        _res = ((_a*self.mask) + (_b* (self.mask==0)))/2
        return _res.reshape(_a.shape)

    def __getitem__(self, index):
        # Randomly select two images from the original dataset
        index1 = random.randint(0, self.num_images - 1)
        index2 = random.randint(0, self.num_images - 1)
        img1, _ = self.original_dataset[index1]
        img2, _ = self.original_dataset[index2]


        return self.merge_two_images(img1[0], img2[0])

    def __len__(self):
        # Return the number of pairs we can make from the original dataset
        return self.num_images