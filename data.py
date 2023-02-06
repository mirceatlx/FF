import numpy as np
import torch
import torch.nn as nn



class CIFAR10:


    @staticmethod
    def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor):

        x = x.view(-1, 3072) # flatten the image
        overlay = torch.zeros(x.shape[0], 10)
        for i, idx in enumerate(y.detach().numpy()): # TODO: get rid of for loop
            overlay[i][idx] = torch.Tensor([1])
        x[:, :10] = overlay 
        return x, y

    @staticmethod
    def predict(data_loader: torch.utils.data.DataLoader, model: nn.Module):

        model.eval()
        for i, (x, y) in enumerate(data_loader):
            for num in range(10):
                over_x = CIFAR10.overlay_y_on_x(x, y)
                out = model(over_x)
                print(out.shape)
                # TODO: finish 






        

