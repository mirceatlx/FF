import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10:

    @staticmethod
    def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor):

        # torch.view acts as a pointer 
        _data = x.clone()
        # pad the image with zeros for the label
        _data = F.pad(_data, (1, 1, 1, 1), "constant", 0)
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
        mask = mask.view(1, 1, 32, 32)
        for step in range(blurring_steps):
            mask = torch.nn.functional.conv2d(mask, filter, padding="same")

        # threshold the image at 0.5
        mask = mask < 0.5
        mask = mask.int().view(32, 32)
        reversed_mask = torch.bitwise_not(mask.bool()).int().view(32, 32)
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
        
