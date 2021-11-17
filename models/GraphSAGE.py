import torch
from torch_geometric.nn import SAGEConv, global_add_pool
import torch.nn.functional as F


class GraphSAGE(torch.nn.Module):
    '''
    A super casual implementation of GraphSAGE
    http://snap.stanford.edu/graphsage/

    params

        in_channels [int]
            the dimension of vertex feature

        hiddens [list of int]
            dimensions of hidden layers

        out_channels [int]
            the number of classes or regression targets

        dropout [float or None]
            the dropout rate, default is None

    '''
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=None):
        super(GraphSAGE, self).__init__()

        self.embedding_layer = torch.nn.Linear(in_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout

        self.activation = torch.nn.LeakyReLU(0.1)
        #self.activation = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding_layer(x)

        for i in range(len(self.convs)-1):
            x = self.convs[i](x, edge_index)
            x = self.activation(x)
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # last layer, no relu and dropout
        x = self.convs[-1](x, edge_index)

        # global pooling
        x = global_add_pool(x, batch=batch)

        # linears
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x