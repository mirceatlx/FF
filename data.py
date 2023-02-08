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
        
