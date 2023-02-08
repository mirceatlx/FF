import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from ff import FF, FFLayer
from data import MNIST
from tqdm import tqdm



if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"

  seed = 42
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  np.random.seed(seed)

  batch_size_train = 512
  batch_size_test = 512
  print("Loading datasets:")
  train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets/MNIST/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets/MNIST/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

  threshold = 1.5
  epochs = 50
  model = FF(logging=True, device=device)
  model.add_layer(FFLayer(nn.Linear(784, 500), optimizer=torch.optim.Adam, epochs=epochs,
         threshold=threshold, activation=nn.ReLU(), lr=0.001, positive_lr=0.005, negative_lr=0.005, logging=True, name="layer 1"))
  model.add_layer(FFLayer(nn.Linear(500, 500), optimizer=torch.optim.Adam, epochs=epochs,
         threshold=threshold, activation=nn.ReLU(), lr=0.001, positive_lr=0.005, negative_lr=0.005, logging=True, name="layer 2"))
  model.add_layer(FFLayer(nn.Linear(500, 500), optimizer=torch.optim.Adam, epochs=epochs,
         threshold=threshold, activation=nn.ReLU(), lr=0.001, positive_lr=0.005, negative_lr=0.005, logging=True, name="layer 3"))

  epochs = 300
  wandb.init(project="MNIST", entity="ffalgo")
  wandb.config = {
    "learning_rate": 0.005,
    "layer_epochs": 50,
    "epochs": epochs,
    "batch_size": batch_size_train,
    "activation": "relu",
    "positive_lr": 0.005,
    "negative_lr": 0.005,
    "threshold": threshold,
    "optimizer": torch.optim.Adam,
    "device": device
  }

  model = model.to(device)
  best_acc = 0.0
  print("Start training")
  for i in tqdm(range(epochs)):
      if i % 1 == 0:
          predictions, real = MNIST.predict(test_loader, model, device)
          acc = np.sum(predictions == real)/len(real)
          wandb.log({"Accuracy on test data": acc})
          if acc > best_acc:
              best_acc = acc
              torch.save(model.state_dict(), 'best_mnist.ph')
          
      predictions, real = MNIST.predict(train_loader, model, device)
      acc = np.sum(predictions == real)/len(real)
      wandb.log({"Accuracy on train data": acc})
      model.train()
      for i, (x, y) in enumerate(train_loader):
          x_pos, _ = MNIST.overlay_y_on_x(x, y)
          rnd = torch.randperm(x.size(0))
          x_neg, _ = MNIST.overlay_y_on_x(x, y[rnd])
          x_pos, x_neg = x_pos.to(device), x_neg.to(device)
          losses = model(x_pos, x_neg)
          
  wandb.finish()
  print(f"Finished training")

