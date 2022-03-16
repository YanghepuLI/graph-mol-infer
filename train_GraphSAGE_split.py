'''
================================================================================
Args & init
================================================================================
'''
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sdffile', type=str, required=True,
                    help="Path to SDF file used to filter QM9.")
parser.add_argument('-nlayers', type=int, required=True,
                    help="the number of GNN layers.")
parser.add_argument('-hiddensize', type=int, required=True,
                    help="the hidden size.")
parser.add_argument('-nepochs', type=int, required=True,
                    help="the number of epochs.")
parser.add_argument('-lr', type=float, required=True,
                    help="learning rate.")
parser.add_argument('-outputfile', type=str, required=True,
                    help="output filename.")
parser.add_argument('--verbose', action='store_true',
                    help="print details.")
parser.add_argument('--plot', action='store_true',
                    help="do the plot.")

args = parser.parse_args()

if torch.cuda.is_available():
    if args.verbose:
        print('Using CUDA')
    device = 'cuda'
else:
    if args.verbose:
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
    Filter out unwanted targets
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
from datasets.QM9_noskip import QM9_noskip
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

# Config
targets = [1, 3, 7, 11]
batch_size = 2048
random_seed = 42

if args.verbose:
    print('-----------------')
    print('Loading dataset.')

# the path should locate your QM9 dataset
dataset = QM9_noskip(root='/home/featurize/data/QM9/')

# Select targets
dataset.data.y = dataset.data.y[:, targets]

# Normalize target values
y_mean = dataset.data.y.mean(dim=0)
y_std = dataset.data.y.std(dim=0)
dataset.data.y -= y_mean
dataset.data.y /= y_std


'''
================================================================================
Dataset split
================================================================================
'''
mol_name_list = []
with open(args.sdffile, 'r') as f:
    next_line_is_name = True
    for line in f:
        if next_line_is_name:
            mol_name_list.append(line.strip())
            next_line_is_name = False
        if line.startswith('$$$$'):
            next_line_is_name = True

indices = []
pointer = 0
skip_header = True
end_loop = False
skip_this = False
for name in mol_name_list:
    skip_this = False
    while dataset.data.name[pointer] != name:
        # Molecule names have format "gdb_xxx"
        if int(dataset.data.name[pointer][4:]) > int(name[4:]):
            skip_this = True
            break
        pointer += 1
        if pointer >= len(dataset.data.name):
            end_loop = True
            break
    if end_loop:
        break
    if not skip_this:
        indices.append(pointer)

print(len(indices), 'molecules remaining after filtering.')

train_dataset_split = torch.utils.data.Subset(dataset, indices)

test_size = int(len(train_dataset_split) * 0.8)
train_size = len(train_dataset_split) - test_size
train_set, test_set = torch.utils.data.random_split(
    dataset=train_dataset_split,
    lengths=[train_size, test_size],
    generator=torch.Generator().manual_seed(random_seed)
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=test_size)

num_atom_features = dataset.data.x.shape[1]
num_edge_features = dataset.data.edge_attr.shape[1]
num_classes = dataset.data.y.shape[1]

if args.verbose:
    print('Done.')
    print(dataset)
    print('Train size:', train_size)


'''
================================================================================
Load your model here
Also, set proper parameters for your optimizer
================================================================================
'''
from models.GraphSAGE import GraphSAGE
from torch.optim import lr_scheduler

if args.verbose:
    print('-----------------')
    print('Building model.')

model = GraphSAGE(
    in_channels=num_atom_features,
    hidden_channels=args.hiddensize,
    num_layers=args.nlayers,
    out_channels=num_classes
)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

if args.verbose:
    print('Done.')
    print(model)


'''
================================================================================
Visualization
================================================================================
'''
if args.plot:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    # Config
    point_size = 4
    point_color = '#2A52BE'

    plt.ion()
    fig, axs = plt.subplots(1, 4, figsize=(28, 6))
    plt.show(block=False)

    def update_scatter(y_true, y_pred):
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        axs[3].clear()
        axs[0].set_title('ALPHA')
        axs[0].set_xlabel('Ground Truth')
        axs[0].set_ylabel('Prediction')
        axs[1].set_title('LUMO')
        axs[1].set_xlabel('Ground Truth')
        axs[1].set_ylabel('Prediction')
        axs[2].set_title('U0')
        axs[2].set_xlabel('Ground Truth')
        axs[2].set_ylabel('Prediction')
        axs[3].set_title('CV')
        axs[3].set_xlabel('Ground Truth')
        axs[3].set_ylabel('Prediction')
        if len(y_true) != 0:
            axs[0].scatter(y_true[:, 0], y_pred[:, 0], s=point_size, c=point_color)
            axs[1].scatter(y_true[:, 1], y_pred[:, 1], s=point_size, c=point_color)
            axs[2].scatter(y_true[:, 2], y_pred[:, 2], s=point_size, c=point_color)
            axs[3].scatter(y_true[:, 3], y_pred[:, 3], s=point_size, c=point_color)
        axs[0].set_ylim(axs[0].get_xlim())
        axs[1].set_ylim(axs[1].get_xlim())
        axs[2].set_ylim(axs[2].get_xlim())
        axs[3].set_ylim(axs[3].get_xlim())
        plt.pause(0.0001)
        plt.draw()
        plt.pause(0.0001)

    update_scatter([], [])


'''
================================================================================
Do training
================================================================================
'''
from sklearn.metrics import r2_score

if args.verbose:
    print('-----------------')
    print('Start training...')
    print('-')

for epoch in range(args.nepochs):

    model.train()
    sum_loss = 0
    for iter, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    avg_loss = sum_loss / len(train_loader) / num_classes
    scheduler.step()

    model.eval()
    for batch in test_loader:
        batch.to(device)
        out = model(batch)
        y_true = batch.y.cpu().detach().numpy()
        y_pred = out.cpu().detach().numpy()

    if args.plot:
        update_scatter(y_true, y_pred)

    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    n = test_size
    p = num_atom_features + num_edge_features
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)

    mae = np.mean(np.abs(y_pred - y_true), axis=0)

    if args.verbose:
        print('Epoch {:03d} \t Avg. train loss: {:.4f}'.format(epoch, avg_loss))
        print('-')
        print('\t\t ALPHA ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[0], mae[0]))
        print('\t\t LUMO  ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[1], mae[1]))
        print('\t\t U0    ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[2], mae[2]))
        print('\t\t CV    ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[3], mae[3]))
        print('-')


'''
================================================================================
Save checkpoint
================================================================================
'''
# Not now!
#torch.save(model.state_dict(), 'checkpoints/weights.pt')


'''
================================================================================
Write results
================================================================================
'''

print('Dataset size :', len(indices))
print('Num layers   :', args.nlayers)
print('Hidden size  :', args.hiddensize)
print('Epochs       :', args.nepochs)
print('Learning rate:', args.lr)
print('ALPHA ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[0], mae[0]))
print('LUMO  ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[1], mae[1]))
print('U0    ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[2], mae[2]))
print('CV    ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[3], mae[3]))
print()

with open(args.outputfile, "a+") as f:
    f.write('Dataset size : {}\n'.format(len(indices)))
    f.write('Num layers   : {}\n'.format(args.nlayers))
    f.write('Hidden size  : {}\n'.format(args.hiddensize))
    f.write('Epochs       : {}\n'.format(args.nepochs))
    f.write('Learning rate: {}\n'.format(args.nepochs))
    f.write('ALPHA ADJ R2: {:.4f} \t MAE: {:.4f}\n'.format(adj_r2[0], mae[0]))
    f.write('LUMO  ADJ R2: {:.4f} \t MAE: {:.4f}\n'.format(adj_r2[1], mae[1]))
    f.write('U0    ADJ R2: {:.4f} \t MAE: {:.4f}\n'.format(adj_r2[2], mae[2]))
    f.write('CV    ADJ R2: {:.4f} \t MAE: {:.4f}\n'.format(adj_r2[3], mae[3]))
    f.write('\n\n')