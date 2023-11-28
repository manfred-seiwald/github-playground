import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import torch
from torch.utils.data import random_split, DataLoader

from captum.attr import IntegratedGradients

from NN_model_architectures import BinaryClassificationWithSigmoid
from NN_datasets import ReactomeDataset
from NN_utils import probability_learner, save_corr, ig_calculator

# ------------ getting AUC values ---------------
directory = "trained_models/drop_out60"
model_AUC = pd.DataFrame()
models = [i for i in range(35)]
AUC = []
for file in os.listdir(directory):
    auc = re.findall(r"\d+\.\d+", file)
    AUC.append(float(auc[0]))

model_AUC["Model"] = models
model_AUC["AUC"] = AUC

model_AUC.to_csv("data/dropout60_AUC.csv", index=False)

# plotting AUV values
no_dropout_auc = pd.read_csv("data/no_dropout_AUC.csv")
no_dropout_auc["Models"] = "no_dropout"
dropout20_auc = pd.read_csv("data/dropout20_AUC.csv")
dropout20_auc["Models"] = "dropout20"
dropout40_auc = pd.read_csv("data/dropout40_AUC.csv")
dropout40_auc["Models"] = "dropout40"
dropout50_auc = pd.read_csv("data/dropout50_AUC.csv")
dropout50_auc["Models"] = "dropout50"
dropout60_auc = pd.read_csv("data/dropout60_AUC.csv")
dropout60_auc["Models"] = "dropout60"
dropout80_auc = pd.read_csv("data/dropout80_AUC.csv")
dropout80_auc["Models"] = "dropout80"

models_auc = pd.concat([no_dropout_auc, dropout20_auc, dropout40_auc, dropout50_auc, dropout60_auc, dropout80_auc], ignore_index=True, sort=False, axis=0)
fig = sns.boxplot(models_auc, x="Models", y="AUC")
plt.title("AUC of models")
fig.set(ylabel="validation_AUC")
save_corr("model_AUC")





