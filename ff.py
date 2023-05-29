import torch
from FFLayer import FFLayer

class FF(torch.nn.Module):
    """
    A Forward Forward network composed of multiple FFLayer instances.

    Args:
        device (str): The device to run the network on. Default is "cpu".

    Attributes:
        layers (torch.nn.ModuleList): A list of FFLayer instances.
        device (torch.device): The device to run the network on.

    Methods:
        add_layer: Adds a new FFLayer instance to the network.
        forward_positive: Computes the forward pass for the positive input.
        forward_negative: Computes the forward pass for the negative input.
        forward: Computes the forward pass for both positive and negative inputs.
        call: Returns the goodness of an input.
    """
    def __init__(self, device="cpu"):
        """
        Initializes the FF instance.
        """
        super(FF, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.device = torch.device(device)

    def add_layer(self, N, M, learning_rate, learning_rate_pos, learning_rate_neg, threshold, epochs):
        """
        Adds a new FFLayer instance to the network.
        
        Args:
            N (int): The number of input neurons.
            M (int): The number of output neurons.
            learning_rate (float): The learning rate for the optimizer.
            learning_rate_pos (float): The learning rate for the positive optimizer.
            learning_rate_neg (float): The learning rate for the negative optimizer.
            threshold (float): The threshold for the goodness function.
            epochs (int): The number of epochs to train the layer for.
        """
        layer = FFLayer(N, M, learning_rate, learning_rate_pos, learning_rate_neg, threshold, epochs).to(self.device)
        self.layers.append(layer)
        self.layers.to(self.device)

    def forward_positive(self, x_pos: torch.Tensor):
        """
        Computes the forward pass for the positive input.
        
        Args:
            x_pos (torch.Tensor): The positive input.
        
        Returns:
            torch.Tensor: The loss for the positive input.
        """
        losses = []
        for layer in self.layers:
            x_pos, loss = layer.forward_positive(x_pos)
            losses.append(loss)
        return torch.mean(torch.tensor(losses))

    def forward_negative(self, x_neg: torch.Tensor):
        """
        Computes the forward pass for the negative input.
        
        Args:
            x_neg (torch.Tensor): The negative input.
        
        Returns:
            torch.Tensor: The loss for the negative input.
        """
        losses = []
        for layer in self.layers:
            x_neg, loss = layer.forward_negative(x_neg)
            losses.append(loss)
        return torch.mean(torch.tensor(losses))

    def forward(self, x_pos: torch.Tensor, x_neg: torch.Tensor = None):
        """
        Computes the forward pass for both positive and negative inputs.

        Args:
            x_pos (torch.Tensor): The positive input.
            x_neg (torch.Tensor): The negative input. Default is None.
        
        Returns:
            torch.Tensor: The output tensor if x_neg is None, otherwise the loss tensor.
        """
        losses = []
        if x_neg is None:
            for layer in self.layers:
                x_pos, _ = layer.forward_positive(x_pos)
            return x_pos

        for layer in self.layers:
            x_pos, x_neg, loss = layer.forward_with_loss(x_pos, x_neg)
            losses.append(loss)
        return torch.mean(torch.tensor(losses))

    def call(self, x):
        """
        Returns the goodness of an input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The goodness of the input.
        """
        goodness = torch.zeros(x.shape[0]).to(self.device)
        for layer in self.layers:
            x = layer.forward(x)
            goodness += layer.goodness_function(x)
        return goodness
