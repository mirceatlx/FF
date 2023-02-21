import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from ff import FF, FFLayer, FFEncoder
from data import MNIST, MergedDataset
from tqdm import tqdm

# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"


batch_size_train = 512
batch_size_test = 512


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets/MNIST/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
train_loader_negative = torch.utils.data.DataLoader(MergedDataset(torchvision.datasets.MNIST('./datasets/MNIST/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))), batch_size=batch_size_train, shuffle=True)


test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets/MNIST/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)



squared_error = lambda x: x.pow(2).mean(1)
deviation_error = lambda x: -((x - x.mean(1).unsqueeze(1)).pow(2).mean(1))


threshold = 1.5
epochs_per_layer = 50
model = FF(logging=False, device=device)
optim_config = {
    "lr": 0.001,
}
positive_optim_config = {
    "lr": 0.001,

}
negative_optim_config = {
    "lr": 0.001,
}

goodness_function = squared_error
awake_period = 32
sleep_period = 32

model.add_layer(FFLayer(nn.Linear(784, 500).to(device), optimizer=torch.optim.Adam, epochs=epochs_per_layer, threshold=threshold, activation=nn.ReLU(), optim_config=optim_config, positive_optim_config=positive_optim_config, negative_optim_config=negative_optim_config, logging=False, name="layer 1", device = device, goodness_function=goodness_function))
model.add_layer(FFLayer(nn.Linear(500, 500).to(device), optimizer=torch.optim.Adam, epochs=epochs_per_layer, threshold=threshold, activation=nn.ReLU(), optim_config=optim_config, positive_optim_config=positive_optim_config, negative_optim_config=negative_optim_config, logging=False, name="layer 2", device = device, goodness_function=goodness_function))
model.add_layer(FFLayer(nn.Linear(500, 500).to(device), optimizer=torch.optim.Adam, epochs=epochs_per_layer, threshold=threshold, activation=nn.ReLU(), optim_config=optim_config, positive_optim_config=positive_optim_config, negative_optim_config=negative_optim_config, logging=False, name="layer 3", device = device, goodness_function=goodness_function))

# wandb.init(project="MNIST", entity="ffalgo", name="autoencoder-fixed_negative_data-32-awake-32-sleep", settings=wandb.Settings(start_method="thread"))
# wandb.config = {
#   "learning_rate": 0.01,
#   "awake_period": awake_period,
#   "sleep_period": sleep_period,
#   "epochs_per_layer": epochs_per_layer,
#   "batch_size": 512,
#   "activation": "relu",
#   "positive_lr": 0.0001,
#   "negative_lr": 0.0001,
#   "threshold": threshold,
#   "optimizer": torch.optim.Adam,
#   "device": device
# }



model = model.to(device)
epochs = 500
best_acc = 0.0
hour = 0
def get_random_number_besides(x):
    import random
    num = random.randint(0,9)
    if num==x: return get_random_number_besides(x)
    return num
def get_negative_y(y):
    return torch.tensor([get_random_number_besides(i) for i in y], dtype = torch.long).to(device)

train_accs = []
test_accs = []

for i in tqdm(range(epochs)):
    if i % 4 == 1:
        predictions, real = MNIST.predict(test_loader, model, device)
        acc = np.sum(predictions == real)/len(real)
        #wandb.log({"Accuracy on test data": acc})
        train_accs.append(acc)
        if acc > best_acc and acc > 0.8:
            best_acc = acc
            # torch.save(model.state_dict(), 'best_mnist_80%.ph')
        
    predictions, real = MNIST.predict(train_loader, model, device)
    acc = np.sum(predictions == real)/len(real)
    #wandb.log({"Accuracy on train data": acc})
    test_accs.append(acc)
    model.train()
    for _, (x, y) in enumerate(train_loader):
        x_pos, _ = MNIST.overlay_y_on_x(x, y)
        x_neg, _ = MNIST.overlay_y_on_x(x, get_negative_y(y))
        x_pos, x_neg = x_pos.to(device), x_neg.to(device)
        if hour % (awake_period + sleep_period) < awake_period:
            model.forward_positive(x_pos)
        else:
            model.forward_negative(x_neg)
        # model.forward(x_pos, x_neg)
        
        hour += 1

train_accs = np.array(train_accs)
test_accs = np.array(test_accs)       

np.save(f"normal-fixed_negative_data-{awake_period}-awake-{sleep_period}-sleep-train", train_accs)
np.save(f"normal-fixed_negative_data-{awake_period}-awake-{sleep_period}-sleep-test", test_accs)
#wandb.finish()
