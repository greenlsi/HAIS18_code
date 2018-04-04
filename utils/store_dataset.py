import pandas as pd
import numpy as np

# Import a csv dataset as pandas array using pandas.
def importCSVasPandas(filePath, delim, force=False):
    dataset = pd.read_csv(filePath,sep=delim,header=-1, low_memory=False)
    return dataset

def randomize(dataset, labels, multi=None):
    np.random.seed(123)
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation, :]
    if multi is not None:
        shuffled_multi = multi[permutation, :]
        return shuffled_dataset, shuffled_labels, shuffled_multi
    else:
        return shuffled_dataset, shuffled_labels