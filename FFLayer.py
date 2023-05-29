import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import no_grad


class FFLayer(nn.Module):
    """
    A Forward Forward layer.

    Args:
        N (int): The number of input features.
        M (int): The number of output features.
        learning_rate (float): The learning rate for the optimizer.
        learning_rate_pos (float): The learning rate for the positive optimizer.
        learning_rate_neg (float): The learning rate for the negative optimizer.
        threshold (float): The threshold for the goodness function.
        epochs (int): The number of epochs to train the layer.

    Attributes:
        linear (torch.nn.Linear): The linear layer.
        optimizer (torch.optim.Adam): The optimizer.
        optimizer_pos (torch.optim.Adam): The positive optimizer.
        optimizer_neg (torch.optim.Adam): The negative optimizer.
        threshold (float): The threshold for the goodness function.
        epochs (int): The number of epochs to train the layer.

    Methods:
        forward: Computes the forward pass for the layer.
        goodness_function: Computes the goodness for the layer activations.
        forward_positive: Computes the forward pass for the positive input.
        forward_negative: Computes the forward pass for the negative input.
        forward_with_loss: Computes the forward pass for both positive and negative inputs.
    """
    def __init__(self, N, M, learning_rate, learning_rate_pos, learning_rate_neg, threshold, epochs):
        super().__init__()

        self.linear = nn.Linear(N, M)
        self.optimizer = optim.Adam(self.linear.parameters(), learning_rate)
        self.optimizer_pos = optim.Adam(self.linear.parameters(), learning_rate_pos)
        self.optimizer_neg = optim.Adam(self.linear.parameters(), learning_rate_neg)
        self.threshold = threshold
        self.epochs = epochs

    def forward(self, x):
        """
        Computes the forward pass for the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x_norm = torch.linalg.norm(x, 2.0, dim=1, keepdim=True)
        x_norm += 1e-4
        x_dir = x / x_norm
        return F.relu(self.linear(x_dir))

    def goodness_function(self, x):
        """
        Computes the goodness for the layer activations.

        Args:
            x (torch.Tensor): The activations of the layer.

        Returns:
            torch.Tensor: The goodness result.
        """
        return x.pow(2).mean(1)

    def forward_positive(self, x_pos):
        """
        Computes the forward pass for the positive input.

        Args:
            x_pos (torch.Tensor): The positive input tensor.

        Returns:
            Tuple[torch.Tensor, float]: The output tensor and the average loss.
        """
        losses = []
        for _ in range(self.epochs):
            self.optimizer_pos.zero_grad()
            pos_good = self.goodness_function(self.forward(x_pos))
            loss = F.softplus(-pos_good + self.threshold).mean()
            losses.append(loss.item())
            loss.backward()
            self.optimizer_pos.step()

        with no_grad():
            h_pos = self.forward(x_pos)

        return h_pos, sum(losses) / len(losses)

    def forward_negative(self, x_neg):
        """
        Computes the forward pass for the negative input.

        Args:
            x_neg (torch.Tensor): The negative input tensor.

        Returns:
            Tuple[torch.Tensor, float]: The output tensor and the average loss.
        """
        losses = []
        for _ in range(self.epochs):
            self.optimizer_neg.zero_grad()
            neg_good = self.goodness_function(self.forward(x_neg))
            loss = F.softplus(neg_good - self.threshold).mean()
            losses.append(loss.item())
            loss.backward()
            self.optimizer_neg.step()

        with no_grad():
            h_neg = self.forward(x_neg)

        return h_neg, sum(losses) / len(losses)

    def forward_with_loss(self, x_pos, x_neg):
        """
        Computes the forward pass for both positive and negative inputs.

        Args:
            x_pos (torch.Tensor): The positive input tensor.
            x_neg (torch.Tensor): The negative input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]: The output tensors and the average loss.
        """
        if x_neg is None:
            return self.forward(x_pos), None, 0.0

        losses = []
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            pos_good = self.goodness_function(self.forward(x_pos))
            neg_good = self.goodness_function(self.forward(x_neg))
            pos_loss = F.softplus(-pos_good + self.threshold)
            neg_loss = F.softplus(neg_good - self.threshold)
            loss = torch.cat([pos_loss, neg_loss]).mean()
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

        with no_grad():
            h_pos = self.forward(x_pos)
            h_neg = self.forward(x_neg)

        return h_pos, h_neg, sum(losses) / len(losses)