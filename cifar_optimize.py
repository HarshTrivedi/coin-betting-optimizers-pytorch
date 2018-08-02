import math
import os
import shutil
import copy

from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, transforms

import cocob


######## Configs ##############
epochs           = 50
batch_size       = 100
test_batch_size  = 100
cuda             = torch.cuda.is_available()
##############################

torch.manual_seed(1)
print('Cuda Availibility: {}'.format(cuda))


transform = transforms.Compose(
                       [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='cifar_data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='cifar_data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
if cuda:
    model.cuda()


def train(optimizer, epoch):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    return epoch_loss


def compute_epoch_loss():
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        epoch_loss += loss.detach().item()
    return epoch_loss


name_optimizers = [
            ('cocob-backprop', cocob.CocobBackprop(model.parameters())),
            ('cocob-ons', cocob.CocobOns(model.parameters())),
            ('adam', torch.optim.Adam(model.parameters(), lr=0.01)),
            ('adagrad', torch.optim.Adagrad(model.parameters(), lr=0.001))
        ]


log_losses_dict = {}
initial_model_wts = copy.deepcopy(model.state_dict())
for opt_name, optimizer in name_optimizers:

    print("Using Optimizer: {}".format(opt_name))
    model.load_state_dict(copy.deepcopy(initial_model_wts))

    losses = []
    # At 0th epoch they should start at exact same
    train_loss = compute_epoch_loss()
    losses.append(math.log10(train_loss))
    for epoch in tqdm(range(epochs)):
        train(optimizer, epoch)
        train_loss = compute_epoch_loss()
        losses.append(math.log10(train_loss))
        # test() # we are interested in train result only.
        # since it's optimization.
    log_losses_dict[opt_name] = losses


### This section plots matplotlib plots                       ####
### of log-losses and saves in appropriate directory          ####
#######################################
import matplotlib
matplotlib.use('Agg') # remove if gives error
import matplotlib.pyplot as plt

for opt_name, optimizer in name_optimizers:
    plt.plot( range(len(log_losses_dict[opt_name])), log_losses_dict[opt_name], '-', label=opt_name )
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Log-Loss')
plt.title('Cifar')
# plt.show()
plt.savefig('log-losses-cifar.png')


# ### This section logs tensorboard events                      ####
# ### of log-losses and saves in appropriate directory          ####
# ########################################
print('Starting to write to TensorBoard')
vis_dir_name = 'tensorboard_vis_cifar'
if os.path.isdir(vis_dir_name):
    shutil.rmtree(vis_dir_name)
writer = SummaryWriter(vis_dir_name)

for t in tqdm(range(len((list(log_losses_dict.values()))[0]))):
    iterate_data_dict = { key: log_losses_dict[key][t]
                        for key in log_losses_dict.keys() }
    writer.add_scalars('log_losses', iterate_data_dict , t)

# Run the following in terminal:
# tensorboard --logdir tensorboard_vis_random_neural
# ########################################