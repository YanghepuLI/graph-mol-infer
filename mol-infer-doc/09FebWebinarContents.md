## Today's Sum-up
+ Comments to our graph learning model
+ Limitations on existing comparison results
+ Explore what next to be done

## Graph Learning Model
+ Current model cannot be called GraphSAGE.
+ We need to give a detailed explanation on the model architecture (including conv layers, dropouts, activations, hidden dimensions, number of params, etc.).
+ Prepare some explanatory figure for future discussions.

## Training & Testing
+ Train set and test set are not consistent with mol-infer.
+ Adjust the train set and test set, keep them the same as what is fed to Mol-infer.
+ Use 5-fold cross-validation.
+ Give results on both train set and test set, check about the over-fitting issue.
+ Draw a figure that shows how mol-infer and GNN perform as train size grow from about 500 to 20,000 molecules.
+ Use random seeds to control the generation of random numbers.

## More Datasets
+ Try more properties and more datasets.
+ Check other datasets used in mol-infer.
+ Maybe we need to deal with sdf files.

## Future Direction
+ Some descriptors in mol-infer cannot be learned by current graph learning methods (the frequency of fringe-trees).
+ If adding these descriptors to the initial node features helps?
+ Explore graph learning's representations, how to evaluation them and explain them?
