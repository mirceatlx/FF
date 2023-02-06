import torch 
import numpy as np
import torch.nn as nn
import torch.functional as F

# reference: https://keras.io/examples/vision/forwardforward/

class FFLayer(nn.Module):

    """
    """

    def __init__(self, in_dims: int, out_dims: int, threshold: float, epochs: int, optimizer: torch.optim):

       super(FFLayer, self).__init__()

       self.in_dims = in_dims
       self.out_dims = out_dims
       self.threshold = threshold
       self.epochs = epochs

       self.linear = nn.Linear(in_dims, out_dims) 
       self.optim = optimizer(self.parameters()) # does this work?


    def forward(self, x: torch.Tensor):

        # normalization: keep the direction, remove any trace of goodness from the last layer

        x_norm = torch.linalg.norm(x, ord=2, dim=1, keepdims=True) 
        x_norm = x_norm + 1e-4
        x_dir = x / x_norm
        out = self.linear(x_dir)

        return F.relu(out)

    def forward(self, x_pos: torch.Tensor, x_neg: torch.Tensor):
        
        losses = []
        for i in range(self.epochs):
            self.optim.zero_grad()
            pos_good = self.goodness(self.forward(x_pos))
            neg_good = self.goodness(self.forward(x_neg))

            loss = torch.log(1 + torch.exp(
                torch.cat([-pos_good + self.threshold, neg_good - self.threshold])))

            mloss = torch.mean(loss)
            losses.append(mloss.detach().numpy())
            mloss.backward()
            self.optim.step()

        with torch.no_grad():

            h_pos = self.forward(x_pos)
            h_neg = self.forward(x_neg)

            return h_pos, h_neg, np.mean(losses)





    def goodness(self, x: torch.Tensor):
        """
        """
        return torch.mean(torch.pow(x, 2), dim=1)


class FF(nn.Module):
    
    """
    """

    def __init__(self, num_layers: int, config: dict(), optimizer: torch.optim):

        super(FF, self).__init__()

        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            layers.append(FFLayer(config["in_dims"][i], config["out_dims"][i], 
                                  config["threshold"], config["epochs"], 
                                  optimizer))


    def forward(self, x_pos, x_neg):

       losses = []
       for i, layer in enumerate(self.layers):

           x_pos, x_neg, loss = layer(x_pos, x_neg)
           losses.append(loss)

       return np.mean(losses)

        
