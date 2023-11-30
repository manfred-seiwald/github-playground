from typing import cast, Tuple
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split
from NN_datasets import ReactomeDataset
import torch
from NN_model_architectures import BinaryClassificationWithDropoutSmall, BinaryClassificationWithSigmoid, BinaryClassificationWithDropout
from NN_utils import ig_calculator

# --------------- data ---------------
# test_df = pd.read_csv("data/ig_features.csv")
# test_features = np.array(test_df)
# test_features_tensor = torch.from_numpy(test_features).type(torch.FloatTensor)  # making tensor
# test_features_tensor.requires_grad_()

dataset = ReactomeDataset()

train_size = int(0.8 * len(dataset))  # 80% of the dataset
test_size = len(dataset) - train_size

trainset, testset = random_split(dataset, [train_size, test_size])

# batchsize is the whole dataset
trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
testloader = DataLoader(testset, batch_size=len(testset), shuffle=False) 

# ------------- calculating mean attributes ------------

train_features, _ = next(iter(trainloader))
test_features, _ = next(iter(testloader))

train_features = train_features.to(torch.float32)
test_features = test_features.to(torch.float32)

train_features.requires_grad_()
test_features.requires_grad_()

# attr_df = ig_calculator("trained_models/dropout60", BinaryClassificationWithDropout(dropout=0.6), test_features)
# attr_df.to_csv("data/dropout60_attr.csv", index=False)

attr_df = ig_calculator("trained_models/dropout60small", BinaryClassificationWithDropoutSmall(dropout=0.6), test_features)
attr_df.to_csv("data/dropout60small_attr.csv", index=False)

#attr_df = pd.read_csv("data/dropout60_attr.csv")
# correlation is between columns
corr = attr_df.T.corr()
print(f'the mean corr is {np.mean(corr)}')
# compute intersection of top n genes of all runs
n = 20
intersect = set()
genes = np.array(attr_df.columns)
for i in range(0, len(attr_df)):
    attr = np.array(attr_df.iloc[i])
    # sort descending
    sort_idxs = np.argsort(attr)[::-1]
    top_genes = genes[sort_idxs[0:n]]
    if i == 0:
        intersect = set(top_genes)
    else:
        intersect = intersect & set(top_genes) 
print(f'{len(attr_df)} runs have {len(intersect)} common genes out of the top {n} genes of each run')               
print(f'the common genes are: {intersect}')
pass

















