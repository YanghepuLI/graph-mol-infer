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
    def __init__(self, in_channels, embedding_size, hiddens, out_channels, dropout=None):
        super(GraphSAGE, self).__init__()

        self.embedding_layer = torch.nn.Linear(in_channels, embedding_size)

        self.convs = torch.nn.ModuleList()
        if len(hiddens) != 0:
            self.convs.append(SAGEConv(embedding_size, hiddens[0]))
            for i in range(len(hiddens)-1):
                self.convs.append(SAGEConv(hiddens[i], hiddens[i+1]))
            self.convs.append(SAGEConv(hiddens[-1], out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, out_channels))

        self.linear1 = torch.nn.Linear(out_channels, out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding_layer(x)

        for i in range(len(self.convs)-1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # last layer, no relu and dropout
        x = self.convs[-1](x, edge_index)

        # global pooling
        x = global_add_pool(x, batch=batch)

        # linears
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x