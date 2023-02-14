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
                 lr: float = 0.01, positive_lr: float = None, negative_lr: float = None,
                 logging=False, name="layer", device="cpu"):
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
        self.optim = optimizer(self.parameters(), lr=lr)
        self.optim_pos = optimizer(self.parameters(), lr=lr if positive_lr is None else positive_lr)
        self.optim_neg = optimizer(self.parameters(), lr=lr if negative_lr is None else negative_lr)
        self.logging = logging
        self.name = name
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
            pos_good = self.goodness(self.call(x_pos))
            loss = torch.log(torch.add(1, torch.exp(torch.add(-pos_good, self.threshold)))).mean()
            losses.append(loss.item())
            loss.backward()
            self.optim_pos.step()
        with torch.no_grad():
            h_pos = self.call(x_pos)
        if self.logging:
            wandb.log({f"positive data loss on {self.name}": np.mean(losses)})
        return h_pos, np.mean(losses)

    def forward_negative(self, x_neg: torch.Tensor):
        """
        Forward pass for negative data.
        """
        losses = []
        for i in range(self.epochs):
            self.optim_neg.zero_grad()
            neg_good = self.goodness(self.call(x_neg))
            loss = torch.log(torch.add(1, torch.exp(torch.add(neg_good,  -self.threshold)))).mean()
            losses.append(loss.item())
            loss.backward()
            self.optim_neg.step()
        with torch.no_grad():
            h_neg = self.call(x_neg)
        if self.logging:
            wandb.log({f"negative data loss on {self.name}": np.mean(losses)})
        return h_neg, np.mean(losses)

    def forward(self, x_pos: torch.Tensor, x_neg: torch.Tensor = None):
        if x_neg == None:
            return self.call(x_pos)
        losses = []
        for i in range(self.epochs):
            self.optim.zero_grad()
            pos_good = self.goodness(self.call(x_pos))
            neg_good = self.goodness(self.call(x_neg))
            loss = torch.log(torch.add(1, torch.exp(
                torch.cat([torch.add(-pos_good, self.threshold), torch.add(neg_good, -self.threshold)])))).mean()
            losses.append(loss.item())
            loss.backward()
            self.optim.step()

        # disable gradient calculation to allow the result to be passed to the next layer
        with torch.no_grad():
            h_pos = self.call(x_pos)
            h_neg = self.call(x_neg)
        if self.logging:
            wandb.log({f"loss on {self.name}": np.mean(losses)})
        return h_pos, h_neg, np.mean(losses)

    def goodness(self, x: torch.Tensor):
        """
        Goodness of data as specified in the FF paper.
        """
        return x.pow(2).mean(1)


class FF(nn.Module):
    """
    The Forward-Forward algorithm.
    """
    def __init__(self, logging=False, device="cpu"):

        super(FF, self).__init__()
        self.num_layers = 0
        self.logging = logging
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
        if self.logging:
            wandb.log({"overall loss": np.mean(losses)})
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
           goodness += layer.goodness(x)
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

class FFNLP(nn.Module):


    def __init__(self, input_tokens: int, vocab_size: int, n_layers: int, epochs: int):

        super(FFNLP, self).__init__()
        self.input_tokens = input_tokens
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.epochs = epochs
        self.token2emb = FFLayer(nn.Linear, name="token2emb")
        layers = []
        for i in range(n_layers):
            layers.append(FFLayer(nn.Linear, name=f"layer {i}"))
        self.layers = nn.ModuleList(layers)
        self.emb2token = FFLayer(layer=nn.Linear, activation=nn.Sigmoid)
        self.emb2token = LinearSoftmax(self.n_layers * hidden_size, vocab_size, name="emb2token")

    def forward(self, x):
        x = self.token2emb(x)
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)

        x = torch.cat(activations, dim=1)
        x = self.emb2token(x)
        return x

    def train(x: torch.Tensor, prev: torch.Tensor, labels: torch.Tensor, theta: float):

        h_pos, h_neg, loss = self.token2emb(x, prev)
        predictions = []
        for layer in self.layers:
            h_pos, h_neg, loss = layer(h_pos, h_neg)
            predictions.append(torch.cat([h_pos, h_neg], dim=0)) 

        predictions = torch.Tensor(predictions)
        x = torch.cat(predictions, dim=1)
        for i in range(self.epochs):
            x = self.emb2token(x)
        h_pos, h_neg, loss = self.emb2token(x)
