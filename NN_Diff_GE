# simple form of differential gene expression
# for each gene: compute mean of TCR=0 expressions and mean of TCR=1 expression
# find genes with highest differences between means

from typing import cast, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from NN_datasets import ReactomeDataset

dataset = ReactomeDataset()

# batchsize is the whole dataset
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# ------------- calculating mean attributes ------------

features, labels = next(iter(dataloader))
features = features.to(torch.float).numpy()

# TPM same sum expr for each sample
features = features / np.sum(features, axis=1, keepdims=True) * 10**6
# try log2 normalized
features = np.log2(features + 1)


features_0 = features[labels == 0]
features_1 = features[labels == 1]

# mean for each gene
means_0 = np.mean(features_0, axis=0)
means_1 = np.mean(features_1, axis=0)

diffs = np.abs(means_0 - means_1)
genes = np.array(dataset.feature_names)

sorted_idxs = np.argsort(diffs)[::-1]
sorted_diffs = diffs[sorted_idxs]
sorted_genes = genes[sorted_idxs]

print(f'top genes: {sorted_genes[0:20]}')
print(f'top diffs: {sorted_diffs[0:20]}')
pass
