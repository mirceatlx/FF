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
    def predict(data_loader: torch.utils.data.DataLoader, model: nn.Module):

        model.eval()
        for i, (x, y) in enumerate(data_loader):
            for num in range(10):
                over_x = CIFAR10.overlay_y_on_x(x, y)
                out = model(over_x)
                print(out.shape)
                # TODO: finish 



class MNIST:

     @staticmethod
     def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor):

        _data = x.clone()
        _data = _data.view(-1, _data.numel() // _data.shape[0]) # flatten the image
        overlay = torch.zeros(_data.shape[0], 10)
        overlay[range(_data.shape[0]), y] = x.max()
        _data[:, :10] = overlay 
        return _data, y


