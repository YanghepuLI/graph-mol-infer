import os
import numpy as np

from joblib import Parallel, delayed
from tensorflow.keras.utils import get_file
from tqdm import tqdm

from spektral.data import Dataset, Graph
from spektral.utils import label_to_one_hot, sparse
from spektral.utils.io import load_csv, load_sdf

ATOM_TYPES = [1, 6, 7, 8, 9]
BOND_TYPES = [1, 2, 3, 4]

class QM9(Dataset):

    ## This class is modified from Spektral to read sdf file in local enviornment

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"

    def __init__(self, amount = None, n_jobs = 1, **kwargs):
        self.amount = amount
        self.n_jobs = n_jobs
        super().__init__(**kwargs)

    def download(self):
        get_file(
            "qm9.tar.gz",
            self.url,
            extract = True,
            cache_dir = self.path,
            cache_subdir = self.path,
        )
        os.remove(os.path.join(self.path, "qm9.tar.gz"))

    def read(self):
        print("loading QM9 dataset.")
        #load  dir and file_name
        sdf_file = os.path.join("/home/liyang/Documents/csprogram/graph-mol-infer-main/sdf_files", "gdb9_15794_eli.sdf")
        data = load_sdf(sdf_file, amount = self.amount)

        def read_mol(mol):
            x = np.array([atom_to_feature(atom) for atom in mol["atoms"]])
            a, e = mol_to_adj(mol)
            return x, a ,e

        data = Parallel(n_jobs=self.n_jobs)(
            delayed(read_mol)(mol) for mol in tqdm(data, ncols=80)
        )
        x_list, a_list, e_list = list(zip(*data))

        #load labels
        labels_file = os.path.join(self.path, "gdb9.sdf.csv")
        labels = load_csv(labels_file)
        labels = labels.set_index("mol_id").values
        if self.amount is not None:
            labels = labels[:self.amount]

        return [
            Graph(x=x, a=a, e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, labels)
        ]


def atom_to_feature(atom):
    atomic_num = label_to_one_hot(atom["atomic_num"], ATOM_TYPES)
    coords = atom["coords"]
    charge = atom["charge"]
    iso = atom["iso"]

    return np.concatenate((atomic_num, coords, [charge, iso]), -1)

def mol_to_adj(mol):
    row, col, edge_features = [], [], []
    for bond in mol["bonds"]:
        start, end = bond["start_atom"], bond["end_atom"]
        row += [start, end]
        col += [end, start]
        edge_features += [bond["type"]] * 2

    a, e = sparse.edge_index_to_matrix(
        edge_index=np.array((row, col)).T,
        edge_weight=np.ones_like(row),
        edge_features=label_to_one_hot(edge_features, BOND_TYPES),
    )

    return a, e


dataset = QM9()
print(dataset)
print(dataset[0])

#Build model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import BatchLoader
from spektral.layers import GlobalSumPool, GraphMasking, GraphSageConv

# Config
learning_rate = 1e-3
epochs = 50
batch_size = 64
# Parameters
F = dataset.n_node_features
S = dataset.n_edge_features
n_out = dataset.n_labels

#train/test split
idxs = np.random.permutation(len(dataset))
split = int(0.8 * len(dataset))
idx_tr, idx_te = np.split(idxs, [split])
dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]


class Net(Model):
    def __init__(self):
        super().__init__()
        self.masking = GraphMasking()
        self.graph_conv = GraphSageConv(64, activation = "relu")
        self.dropout = Dropout(0.5)
        self.global_pool = GlobalSumPool()
        self.dense = Dense(n_out)
    
    def call(self, inputs):
        x, a = inputs
        x = self.masking(x)
        x = self.graph_conv([x, a])
        x = self.dropout(x)
        output = self.global_pool(x)
        output = self.dense(output)

        return output

model = Net()
optimizer = Adam(learning_rate)
model.compile(optimizer=optimizer, loss="mse")

# Fit model
loader_tr = BatchLoader(dataset_tr, batch_size=batch_size, mask=True)
model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, epochs=epochs)

# Evaluate model
print("Testing model")
loader_te = BatchLoader(dataset_te, batch_size=batch_size, mask=True)
loss = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}".format(loss))