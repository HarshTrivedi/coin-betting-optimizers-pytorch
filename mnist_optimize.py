import math
import os
import shutil
import copy

from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('mnist_data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=test_batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

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
        loss = F.nll_loss(output, target)
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
        loss = F.nll_loss(output, target)
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
plt.xlabel('Epochs')
plt.ylabel('Log-Loss')
plt.legend(loc='upper right')
plt.title('Mnist')
# plt.show()
plt.savefig('log-losses-mnist.png')


# ### This section logs tensorboard events                      ####
# ### of log-losses and saves in appropriate directory          ####
# ########################################
print('Starting to write to TensorBoard')
vis_dir_name = 'tensorboard_vis_mnist'
if os.path.isdir(vis_dir_name):
    shutil.rmtree(vis_dir_name)
writer = SummaryWriter(vis_dir_name)

for t in tqdm(range(len(list(log_losses_dict.values())[0]))):
    iterate_data_dict = { key: log_losses_dict[key][t]
                        for key in log_losses_dict.keys() }
    writer.add_scalars('log_losses', iterate_data_dict , t)

# Run the following in terminal:
# tensorboard --logdir tensorboard_vis_random_neural
# ########################################
