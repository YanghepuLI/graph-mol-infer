# Target
We aim to use graph models to get the prediction of molecure's features. Hoping to get better results than an usual RNN. This requires our graph model need to have relatively
powerful abilities to predict graph-level representations.

# Exsisted models

## Node embedding models
Node embedding model only has the ability to give node repressentations. Node embedding networks mostly deal with the unsupervised learning tasks, like node classification or
node clustering task.

## Generative graph models
Not suitable here. Different target tasks.

## Graph Neural Networks
Graph Neural Networks have abilities to tell graph-level features. Here GNN is the most suitable models.

# Focued on graph-level representation GNN
## Graph Pooling
Similar to the AGGREGATE operator, the task of graph pooling can be viewed as a problem of learning over sets. Want to design a pooling function f_p, which maps a set of node embeddings to an embedding z_G that represents the full graph.
+ take a sum(or mean) of the node embeddings
+ a combination of LSTMs and attention to pool the node embeddings

## Graph conarsening approaches
One limitation of the set pooling approaches is that they do not exploit the structure of the graph. One popular strategy to
accomplish this is to perform graph clustering or coarsening as a means to pool the node representations.

## Generalized Message Passing
Although the most popular style of GNN message passing operates largely at the node level, the GNN meassage passing approach can also be generalized to leverage edge and graph-level information at each stage of message passing. During message passing, generate hidden embeddings h(u,v) for each edge in the graph, as well as an embedding h(G) corresponding to the entire graph.
