import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Config
targets = [1, 3, 7, 11]
test_size = 1000

print('-----------------')
print('Loading dataset.')

dataset = QM9(root='./datasets/QM9/', transform=SelectTargets(targets))
train_size = len(dataset) - test_size
train_set, test_set = torch.utils.data.random_split(
    dataset=dataset,
    lengths=[train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=2048, shuffle=True)
test_loader = DataLoader(test_set, batch_size=test_size, shuffle=False)

num_node_features = dataset.data.x.shape[1]
num_classes = len(targets)

print('Done.')
print(dataset)

'''
================================================================================
Load your model here
Also, set proper parameters for your optimizer
================================================================================
'''
from models.GraphSAGE import GraphSAGE

print('-----------------')
print('Building model.')

model = GraphSAGE(
    in_channels=num_node_features,
    hiddens=[64, 64],
    out_channels=num_classes,
    dropout=0.5
)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

print('Done.')
print(model)


'''
================================================================================
Do training
================================================================================
'''
import torch.nn.functional as F
from sklearn.metrics import r2_score

# Config
num_epochs = 50

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
        sum_loss += loss.item()
    avg_loss = sum_loss / len(train_loader) / num_classes

    model.eval()
    for batch in test_loader:
        batch.to(device)
        out = model(batch)
        y_true = batch.y.cpu().detach().numpy()
        y_pred = out.cpu().detach().numpy()
        r2 = r2_score(y_true, y_pred, multioutput='raw_values')

    print('Epoch {:03d} \t Avg. train loss: {:.4f}'.format(epoch, avg_loss))
    print('-')
    print('\t\t ALPHA R2: {:.4f}'.format(r2[0]))
    print('\t\t LUMO  R2: {:.4f}'.format(r2[1]))
    print('\t\t U0    R2: {:.4f}'.format(r2[2]))
    print('\t\t CV    R2: {:.4f}'.format(r2[3]))
    print('-')


'''
================================================================================
Save checkpoint
================================================================================
'''
torch.save(model.state_dict(), 'checkpoints/weights.pt')