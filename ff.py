import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb

# reference: https://keras.io/examples/vision/forwardforward/
class FFLayer(nn.Module):
    """
    One layer of the Forward-Forward algorithm.
    The network assumes each layer is a black box and doesn't know any
    information about the forward pass.
    """
    def __init__(self, layer: nn.Module, threshold: float=2.0, epochs: int=50, 
                 optimizer: torch.optim = torch.optim.Adam, activation: nn.Module = nn.ReLU(),
                 optim_config: dict = None, positive_optim_config: dict = None, negative_optim_config: dict = None,
                name="layer", device="cpu", goodness_function=lambda x: x.pow(2).mean(1)):
        """
        layer: the layer to be wrapped
        threshold: the threshold for the goodness of the data
        epochs: number of epochs to train the layer
        optimizer: the optimizer to use for training the layer
        activation: the activation function to use for the layer
        """
        super(FFLayer, self).__init__()
        self.threshold = threshold
        self.epochs = epochs
        self.layer = layer
        self.optim = optimizer(self.parameters(), **optim_config)
        self.optim_pos = optimizer(self.parameters(), **(positive_optim_config if positive_optim_config is not None else optim_config))
        self.optim_neg = optimizer(self.parameters(), **(negative_optim_config if negative_optim_config is not None else optim_config))
        self.name = name
        self.device = device
        self.goodness_function = goodness_function
        # loading the activation function
        self.activation = activation

    def call(self, x: torch.Tensor):

        # normalization: keep the direction, remove any trace of goodness from the last layer
        x_norm = torch.linalg.norm(x, ord=2, dim=1, keepdims=True) 
        x_norm = torch.add(x_norm, 1e-4)
        x_dir = torch.div(x, x_norm)

        return self.activation(self.layer(x_dir))

    def forward_positive(self, x_pos: torch.Tensor):
        """
        Forward pass for positive data.
        """
        losses = []
        for i in range(self.epochs):
            self.optim_pos.zero_grad()
            pos_good = self.goodness_function(self.call(x_pos))
            loss = torch.log(torch.add(1, torch.exp(torch.add(-pos_good, self.threshold)))).mean()
            losses.append(loss.item())
            loss.backward()
            self.optim_pos.step()
        with torch.no_grad():
            h_pos = self.call(x_pos)
        return h_pos, np.mean(losses)

    def forward_negative(self, x_neg: torch.Tensor):
        """
        Forward pass for negative data.
        """
        losses = []
        for i in range(self.epochs):
            self.optim_neg.zero_grad()
            neg_good = self.goodness_function(self.call(x_neg))
            loss = torch.log(torch.add(1, torch.exp(torch.add(neg_good,  -self.threshold)))).mean()
            losses.append(loss.item())
            loss.backward()
            self.optim_neg.step()
        with torch.no_grad():
            h_neg = self.call(x_neg)
        return h_neg, np.mean(losses)

    def forward(self, x_pos: torch.Tensor, x_neg: torch.Tensor = None):
        if x_neg == None:
            return self.call(x_pos)
        losses = []
        for i in range(self.epochs):
            self.optim.zero_grad()
            pos_good = self.goodness_function(self.call(x_pos))
            neg_good = self.goodness_function(self.call(x_neg))
            loss = torch.log(torch.add(1, torch.exp(
                torch.cat([torch.add(-pos_good, self.threshold), torch.add(neg_good, -self.threshold)])))).mean()
            losses.append(loss.item())
            loss.backward()
            self.optim.step()

        # disable gradient calculation to allow the result to be passed to the next layer
        with torch.no_grad():
            h_pos = self.call(x_pos)
            h_neg = self.call(x_neg)
        return h_pos, h_neg, np.mean(losses)


class FF(nn.Module):
    """
    The Forward-Forward algorithm.
    """
    def __init__(self, device="cpu"):

        super(FF, self).__init__()
        self.num_layers = 0
        self.layers = []
        self.device = device
    
    """
    Add a layer to the network
    """
    def add_layer(self, layer: FFLayer):
        self.layers.append(layer.to(self.device))
        self.num_layers += 1

    """
    Forward pass for positive data.
    """
    def forward_positive(self, x_pos: torch.Tensor):

       losses = []
       for _, layer in enumerate(self.layers):
           x_pos, loss = layer.forward_positive(x_pos)
           losses.append(loss)
       return np.mean(losses)
    
    """
    Forward pass for negative data.
    """
    def forward_negative(self, x_neg: torch.Tensor):
        
       losses = []
       for _, layer in enumerate(self.layers):
           x_neg, loss = layer.forward_negative(x_neg)
           losses.append(loss)
       return np.mean(losses)
    
    def forward(self, x_pos: torch.Tensor, x_neg: torch.Tensor = None):

        losses = []
        if x_neg == None:
            return self.call(x_pos)
        
        for _, layer in enumerate(self.layers):
            x_pos, x_neg, loss = layer(x_pos, x_neg)
            losses.append(loss)
        return np.mean(losses)

    """
    Calculate the total goodness of the network for some data (positive/negative)
    """
    def call(self, x):
       """
       Calculate the total goodness of the network for some data (positive/negative)
       """
       goodness = torch.zeros(x.shape[0]).to(self.device)
       for i, layer in enumerate(self.layers):
           x = layer(x)
           goodness += layer.goodness_function(x)
       return goodness

class LinearSoftmax(nn.Module):


    def __init__(self, in_features: int, out_features: int, loss_fn=nn.NLLLoss, name="layer"):
        self.in_features = in_features
        self.out_features = out_features
        self.loss_fn = loss_fn()
        self.linear = nn.Linear(in_features, out_features)
        self.soft = nn.Softmax(dim=1)
        super(LinearSoftmax, self).__init__()

    def forward(self, x: torch.Tensor):
        x_norm = torch.linalg.norm(x, ord=2, dim=1, keepdims=True) 
        x_norm = torch.add(x_norm, 1e-4)
        x = torch.div(x, x_norm)
        return self.soft(self.linear(x))