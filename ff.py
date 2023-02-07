import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# reference: https://keras.io/examples/vision/forwardforward/
class FFLayer(nn.Module):
    """
    One layer of the Forward-Forward algorithm.
    The network assumes each layer is a black box and doesn't know any
    information about the forward pass.
    """
    def __init__(self, in_dims: int, out_dims: int, threshold: float=2.0, epochs: int=50, 
                 optimizer: torch.optim = torch.optim.Adam):
       """
       in_dims: dimensions of the input for nn.Linear
       out_dims: dimensions of the output given by nn.Linear
       threshold: threshold to differentiate between negative and positive data
       epochs: number of epochs to go through the layer
       optimizer: torch optimizer performing weight updates
       """
       super(FFLayer, self).__init__()
       self.in_dims = in_dims
       self.out_dims = out_dims
       self.threshold = threshold
       self.epochs = epochs
       self.linear = nn.Linear(in_dims, out_dims)
       self.optim = optimizer(self.parameters())

    def call(self, x: torch.Tensor):

        # normalization: keep the direction, remove any trace of goodness from the last layer
        x_norm = torch.linalg.norm(x, ord=2, dim=1, keepdims=True) 
        x_norm = x_norm + 1e-4
        x_dir = x / x_norm
        return F.relu(self.linear(x_dir))

    def forward(self, x_pos: torch.Tensor, x_neg: torch.Tensor = None):
        if x_neg == None:
            return self.call(x_pos)
        losses = []
        for i in range(self.epochs):
            self.optim.zero_grad()
            pos_good = self.goodness(self.call(x_pos))
            neg_good = self.goodness(self.call(x_neg))
            loss = torch.log(1 + torch.exp(
                torch.cat([-pos_good + self.threshold, neg_good - self.threshold]))).mean()
            losses.append(loss.item())
            loss.backward()
            self.optim.step()
      
        # disable gradient calculation to allow the result to be passed to the next layer
        with torch.no_grad():
            h_pos = self.call(x_pos)
            h_neg = self.call(x_neg)
        return h_pos, h_neg, np.mean(losses)

    def goodness(self, x: torch.Tensor):
        """
        Goodness of data as specified in the FF paper.
        """
        return x.pow(2).mean(1)


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

        self.layers = layers


    def forward(self, x_pos: torch.Tensor, x_neg: torch.Tensor = None):

       losses = []
       if x_neg == None:
           return self.call(x_pos)
       for i, layer in enumerate(self.layers):
           x_pos, x_neg, loss = layer(x_pos, x_neg)
           losses.append(loss)
       return np.mean(losses)

    
    def call(self, x):
       """
       Calculate the total goodness of the network for some data (positive/negative)
       """
       goodness = torch.zeros(x.shape[0])
       for i, layer in enumerate(self.layers):
           x = layer(x)
           goodness += layer.goodness(x)

       return goodness

