import numpy as np
import torch

if torch.cuda.is_available():
    print('Using CUDA')
    device = 'cuda'
else:
    print('Using CPU')
    device = 'cpu'

'''
================================================================================
Utils
I put utils here for convenience
================================================================================
'''
from torch_geometric.transforms import BaseTransform

class SelectTargets(BaseTransform):
    '''
    Filters out unwanted targets
    '''
    def __init__(self, targets):
        self.targets = targets

    def __call__(self, data):
        data.y = data.y[:,self.targets]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(targets={self.targets})'


'''
================================================================================
QM9 dataset
Target #1 (ALPHA) #3 (LUMO) #7 (U0) #11 (CV) are used.
Please find:
http://quantum-machine.org/datasets/
https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
https://arxiv.org/pdf/2107.02381.pdf
================================================================================
'''
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

# Config
targets = [1, 3, 7, 11]
test_size = 1000
batch_size = 2048
random_seed = 42

print('-----------------')
print('Loading dataset.')

dataset = QM9(root='./datasets/QM9/')

# Select targets
dataset.data.y = dataset.data.y[:, targets]

# Normalize target values
y_mean = dataset.data.y.mean(dim=0)
y_std = dataset.data.y.std(dim=0)
dataset.data.y -= y_mean
dataset.data.y /= y_std
#dataset.data.y *= 10

train_size = len(dataset) - test_size
train_set, test_set = torch.utils.data.random_split(
    dataset=dataset,
    lengths=[train_size, test_size],
    generator=torch.Generator().manual_seed(random_seed)
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=test_size)

num_node_features = dataset.data.x.shape[1]
num_classes = dataset.data.y.shape[1]

print('Done.')
print(dataset)

'''
================================================================================
Load your model here
Also, set proper parameters for your optimizer
================================================================================
'''
from models.GraphSAGE import GraphSAGE
from torch_geometric.nn import DimeNet
from torch.optim import lr_scheduler

print('-----------------')
print('Building model.')

'''
model = DimeNet(
    hidden_channels=64,
    out_channels=num_classes,
    num_blocks=7,
    num_bilinear=5,
    num_spherical=5,
    num_radial=5
)
'''

model = GraphSAGE(
    in_channels=num_node_features,
    hidden_channels=64,
    num_layers=6,
    out_channels=num_classes
)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.8)

print('Done.')
print(model)


'''
================================================================================
Do training
================================================================================
'''
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.ion()
figure(figsize=(28, 6), dpi=80)

# Config
num_epochs = 100

print('-----------------')
print('Start training...')
print('-')
for epoch in range(num_epochs):

    model.train()
    sum_loss = 0
    for iter, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        sum_loss += loss.item()
    avg_loss = sum_loss / len(train_loader) / num_classes

    model.eval()
    for batch in test_loader:
        batch.to(device)
        out = model(batch)
        y_true = batch.y.cpu().detach().numpy()
        y_pred = out.cpu().detach().numpy()

    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    mae = np.mean(np.abs(y_pred - y_true), axis=0)

    plt.clf()
    plt.subplot(1, 4, 1)
    plt.scatter(y_true[:, 0], y_pred[:, 0])
    plt.subplot(1, 4, 2)
    plt.scatter(y_true[:, 1], y_pred[:, 1])
    plt.subplot(1, 4, 3)
    plt.scatter(y_true[:, 2], y_pred[:, 2])
    plt.subplot(1, 4, 4)
    plt.scatter(y_true[:, 3], y_pred[:, 3])
    plt.pause(0.0001)
    plt.draw()
    plt.show(block=False)
    plt.pause(0.0001)

    print('Epoch {:03d} \t Avg. train loss: {:.4f}'.format(epoch, avg_loss))
    print('-')
    print('\t\t ALPHA R2: {:.4f} \t MAE: {:.4f}'.format(r2[0], mae[0]))
    print('\t\t LUMO  R2: {:.4f} \t MAE: {:.4f}'.format(r2[1], mae[1]))
    print('\t\t U0    R2: {:.4f} \t MAE: {:.4f}'.format(r2[2], mae[2]))
    print('\t\t CV    R2: {:.4f} \t MAE: {:.4f}'.format(r2[3], mae[3]))
    print('-')


'''
================================================================================
Save checkpoint
================================================================================
'''
# Not now!
#torch.save(model.state_dict(), 'checkpoints/weights.pt')