import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torchvision
from torch.utils.data import Dataset


class MNISTInMemory(Dataset):
    """
    A PyTorch dataset for loading the MNIST dataset into memory.

    Args:
        root (str): The root directory where the dataset will be stored. Default is "./datasets/MNIST".
        train (bool): Whether to load the training or testing set. Default is True.

    Attributes:
        mnist_data (torchvision.datasets.MNIST): The MNIST dataset.
        data (List[Tuple[torch.Tensor, int]]): The data as a list of tuples containing an image tensor and its label.

    Methods:
        __getitem__: Gets an item from the dataset.
        __len__: Gets the length of the dataset.
    """
    def __init__(self, root='./datasets/MNIST', train=True):
        self.mnist_data = torchvision.datasets.MNIST(root=root, train=train, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,))
                                                      ]))
        self.data = []
        for img, target in self.mnist_data:
            self.data.append((img, target))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.mnist_data)


class FMNISTInMemory(Dataset):
    """
    A PyTorch dataset for loading the Fashion-MNIST dataset into memory.

    Args:
        root (str): The root directory where the dataset will be stored. Default is "./datasets/FMNIST".
        train (bool): Whether to load the training or testing set. Default is True.

    Attributes:
        fmnist_data (torchvision.datasets.FashionMNIST): The Fashion-MNIST dataset.
        data (List[Tuple[torch.Tensor, int]]): The data as a list of tuples containing an image tensor and its label.

    Methods:
        __getitem__: Gets an item from the dataset.
        __len__: Gets the length of the dataset.
    """
    def __init__(self, root='./datasets/FMNIST', train=True):
        self.fmnist_data = torchvision.datasets.FashionMNIST(root=root, train=train, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,))
                                                      ]))
        self.data = []
        for img, target in self.fmnist_data:
            self.data.append((img, target))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.fmnist_data)


class CIFAR10InMemory(Dataset):
    """
    A PyTorch dataset for loading the CIFAR-10 dataset into memory.

    Args:
        root (str): The root directory where the dataset will be stored. Default is "./datasets/CIFAR10".
        train (bool): Whether to load the training or testing set. Default is True.

    Attributes:
        cifar10_data (torchvision.datasets.CIFAR10): The CIFAR-10 dataset.
        data (List[Tuple[torch.Tensor, int]]): The data as a list of tuples containing an image tensor and its label.

    Methods:
        __getitem__: Gets an item from the dataset.
        __len__: Gets the length of the dataset.
    """
    def __init__(self, root='./datasets/CIFAR10', train=True):
        self.cifar10_data = torchvision.datasets.CIFAR10(root=root, train=train, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                          (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) # mean and std for CIFAR-10
                                                      ]))
        self.data = []
        for img, target in self.cifar10_data:
            self.data.append((img, target))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.cifar10_data)
        
class MergedDataset(Dataset):
    def __init__(self, original_dataset, steps, batch_size, negative_data_generator):
        self.original_dataset = original_dataset
        self.num_images = len(original_dataset)
        self.steps = steps
        self.batch_size = batch_size
        self.negative_data_generator = negative_data_generator

    def __getitem__(self, index):
        imgs1, imgs2 = [], []
        for _ in range(self.batch_size):
            # Randomly select two images from the original dataset
            index1 = random.randint(0, self.num_images - 1)
            index2 = random.randint(0, self.num_images - 1)
            img1, _ = self.original_dataset[index1]
            img2, _ = self.original_dataset[index2]
            imgs1.append(img1)
            imgs2.append(img2)

        # Stack tensors along a new dimension
        imgs1 = torch.stack(imgs1)
        imgs2 = torch.stack(imgs2)

        # Generate negative data
        neg_data = self.negative_data_generator(imgs1, imgs2, self.steps)
        return neg_data

    def __len__(self):
        # Return the number of pairs we can make from the original dataset
        return self.num_images // self.batch_size

class MNIST:

    @staticmethod
    def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor):
        _data = x.clone()
        _data = _data.view(-1, _data.numel() // _data.shape[0])
        overlay = torch.zeros(_data.shape[0], 10)
        overlay[range(_data.shape[0]), y] = x.max()
        _data[:, :10] = overlay 
        return _data, y

    @staticmethod
    def predict(data_loader: DataLoader, model: nn.Module, device="cpu"):
        model.eval()
        predictions = []
        real = []

        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            batch_size = x.shape[0]
            max_goodness = torch.zeros((10, batch_size)).to(device)

            for num in range(10):
                over_x = MNIST.overlay_y_on_x(x.clone(), num)[0]
                good = model.call(over_x)
                max_goodness[num, :] = good.squeeze().detach()

            _, best_nums = torch.max(max_goodness, dim=0)
            predictions.extend(best_nums.cpu().numpy())
            real.extend(y.cpu().numpy())

        return np.array(predictions), np.array(real)

    @staticmethod
    def negative_data_with_masks(a, b, blurring_steps=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        a = a.to(device)
        b = b.to(device)

        mask = torch.rand_like(a) < 0.5
        mask = mask.float()
        filter = torch.tensor([[1/16., 1/8., 1/16.], [1/8., 1/4., 1/8.], [1/16., 1/8., 1/16.]], device=device)
        filter = filter.view(1, 1, 3, 3)

        for _ in range(blurring_steps):
            mask = F.conv2d(mask, filter, padding="same")

        mask = mask < 0.5
        mask = mask.int()
        reversed_mask = torch.bitwise_not(mask.bool()).int()
        neg_data = torch.mul(mask, a) + torch.mul(reversed_mask, b)

        return neg_data
    
    @staticmethod
    def get_in_memory(train):
      return MNISTInMemory(train = train)

    @staticmethod
    def get_image_shape():
      return (1, 28, 28)
class FMNIST:

    @staticmethod
    def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor):
        _data = x.clone()
        _data = _data.view(-1, _data.numel() // _data.shape[0])
        overlay = torch.zeros(_data.shape[0], 10)
        overlay[range(_data.shape[0]), y] = x.max()
        _data[:, :10] = overlay 
        return _data, y

    @staticmethod
    def predict(data_loader: DataLoader, model: nn.Module, device="cpu"):
        model.eval()
        predictions = []
        real = []

        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            batch_size = x.shape[0]
            max_goodness = torch.zeros((10, batch_size)).to(device)

            for num in range(10):
                over_x = FMNIST.overlay_y_on_x(x.clone(), num)[0]
                good = model.call(over_x)
                max_goodness[num, :] = good.squeeze().detach()

            _, best_nums = torch.max(max_goodness, dim=0)
            predictions.extend(best_nums.cpu().numpy())
            real.extend(y.cpu().numpy())

        return np.array(predictions), np.array(real)

    @staticmethod
    def get_in_memory(train):
      return FMNISTInMemory(train = train)

    @staticmethod
    def negative_data_with_masks(a, b, blurring_steps=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        a = a.to(device)
        b = b.to(device)

        mask = torch.rand_like(a) < 0.5
        mask = mask.float()
        filter = torch.tensor([[1/16., 1/8., 1/16.], [1/8., 1/4., 1/8.], [1/16., 1/8., 1/16.]], device=device)
        filter = filter.view(1, 1, 3, 3)

        for _ in range(blurring_steps):
            mask = F.conv2d(mask, filter, padding="same")

        mask = mask < 0.5
        mask = mask.int()
        reversed_mask = ~mask.bool()
        neg_data = mask * a + reversed_mask * b
        return neg_data
    
    @staticmethod
    def get_image_shape():
      return (1, 28, 28)
class CIFAR10:

    @staticmethod
    def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor):
        _data = x.clone()
        _data = _data.view(-1, _data.numel() // _data.shape[0])
        overlay = torch.zeros(_data.shape[0], 10)
        overlay[range(_data.shape[0]), y] = x.max()
        _data[:, :10] = overlay 
        return _data, y    


    @staticmethod
    def predict(data_loader: DataLoader, model: nn.Module, device="cpu"):
        model.eval()
        predictions = []
        real = []

        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            batch_size = x.shape[0]
            max_goodness = torch.zeros((10, batch_size)).to(device)

            for num in range(10):
                over_x = CIFAR10.overlay_y_on_x(x.clone(), num)[0]
                good = model.call(over_x)
                max_goodness[num, :] = good.squeeze().detach()

            _, best_nums = torch.max(max_goodness, dim=0)
            predictions.extend(best_nums.cpu().numpy())
            real.extend(y.cpu().numpy())

        return np.array(predictions), np.array(real)
    
    @staticmethod
    def get_in_memory(train):
      return CIFAR10InMemory(train = train)

    @staticmethod
    def negative_data_with_masks(a, b, blurring_steps=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        a = a.to(device)
        b = b.to(device)

        mask = torch.rand_like(a) < 0.5
        mask = mask.float()
        filter = torch.tensor([[1/16., 1/8., 1/16.], [1/8., 1/4., 1/8.], [1/16., 1/8., 1/16.]], device=device)
        filter = filter.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        for _ in range(blurring_steps):
            mask = F.conv2d(mask, filter, padding="same", groups=3)

        mask = mask < 0.5
        mask = mask.int()
        reversed_mask = ~mask.bool()
        neg_data = mask * a + reversed_mask * b
        return neg_data
    
    @staticmethod
    def get_image_shape():
      return (3, 32, 32)