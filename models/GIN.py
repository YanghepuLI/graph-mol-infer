import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F

class GIN(torch.nn.Module):

    def __init__(self, in_channels, hiddens, mlp_hiddens, out_channels, dropout=None):
        super().__init__()

        if len(hiddens) == 0:
            raise('hiddens cannot be empty')

        self.mlp_hiddens = mlp_hiddens

        self.convs = torch.nn.ModuleList()
        _inner_mlp = self._build_inner_mlp(in_channels, hiddens[0])
        self.convs.append(GINConv(_inner_mlp))
        for i in range(len(hiddens)-1):
            _inner_mlp = self._build_inner_mlp(hiddens[i], hiddens[i+1])
            self.convs.append(GINConv(_inner_mlp))

        self.last_mlp = self._build_inner_mlp(sum(hiddens), out_channels)

        self.dropout = dropout

    def _build_inner_mlp(self, in_channels, out_channels):
        layer_list = []
        layer_list.append(nn.Linear(in_channels, self.mlp_hiddens[0]))
        layer_list.append(nn.ReLU())
        for i in range(len(self.mlp_hiddens)-1):
            layer_list.append(nn.Linear(self.mlp_hiddens[i], self.mlp_hiddens[i+1]))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(self.mlp_hiddens[-1], out_channels))
        return nn.Sequential(*layer_list)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_features = []

        for i in range(len(self.convs)-1):
            x = self.convs[i](x, edge_index)
            x_features.append(x)
            #x = F.relu(x)
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # last layer, no relu and dropout
        x = self.convs[-1](x, edge_index)
        x_features.append(x)

        x_concat = torch.cat(x_features, dim=1)

        # global pooling
        x = global_add_pool(x_concat, batch=batch)

        # linears
        x = self.last_mlp(x)

        return x