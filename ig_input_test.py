import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import re

import torch
from torch.utils.data import random_split, DataLoader

from captum.attr import IntegratedGradients

from NN_model_architectures import BinaryClassificationWithSigmoid
from NN_datasets import ReactomeDataset
from NN_utils import probability_learner, save_corr, upper

# --------- getting data and indices for train test ----------------
dataset = ReactomeDataset()
train_size = int(0.8 * len(dataset))  # 80% of the dataset
test_size = len(dataset) - train_size

# ----------------- calculating IG with random input Tensors  -------------------
rand_results = pd.DataFrame()
directory = "trained_models/no_drop_out"
count = 0
for file in os.listdir(directory):
    count += 1
    print(f"{count}/{len(os.listdir(directory))}")
    trainset, testset = random_split(dataset, [train_size, test_size])

    test_df = pd.DataFrame(columns=dataset.features.columns)
    for i in range(len(testset)):
        test_df.loc[i] = testset[i][0]  # creating df
    test_features = np.array(test_df)
    test_features_tensor = torch.from_numpy(test_features).type(torch.FloatTensor)  # making tensor
    test_features_tensor.requires_grad_()  # last random input tensor is used for identical input

    path_to_file = f"{directory}/{file}"
    model = BinaryClassificationWithSigmoid()
    model.load_state_dict(torch.load(f=path_to_file))

    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(test_features_tensor, return_convergence_delta=True)
    attr = attr.detach().numpy()

    mean_attr = np.mean(attr, axis=0)
    mean_attr_df = pd.DataFrame(data=[mean_attr],
                                columns=dataset.features.columns)

    rand_results = pd.concat([mean_attr_df, rand_results], ignore_index=True)

rand_resultsT = rand_results.T
rand_corr = rand_resultsT.corr()

print("Finished random Input")

# ------------ calculating IG with identical input Tensor ----------

ident_results = pd.DataFrame()
directory = "trained_models/no_drop_out"
count = 0
for file in os.listdir(directory):
    count += 1
    print(f"{count}/{len(os.listdir(directory))}")

    path_to_file = f"{directory}/{file}"
    model = BinaryClassificationWithSigmoid()
    model.load_state_dict(torch.load(f=path_to_file))

    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(test_features_tensor, return_convergence_delta=True)
    attr = attr.detach().numpy()

    mean_attr = np.mean(attr, axis=0)
    mean_attr_df = pd.DataFrame(data=[mean_attr],
                                columns=dataset.features.columns)

    ident_results = pd.concat([mean_attr_df, ident_results], ignore_index=True)

ident_resultsT = ident_results.T
ident_corr = ident_resultsT.corr()
print("Finished identical Input")



# plotting
corr_rand_ident = stats.spearmanr(upper(ident_corr), upper(rand_corr))
print(stats.spearmanr(upper(ident_corr), upper(rand_corr)))

f, axes = plt.subplots(1,2, figsize=(10, 5))
sns.set_style("white")
for ix, m in enumerate([rand_corr, ident_corr]):
    sns.heatmap(m, cmap="coolwarm",center=0.8, vmax=1, vmin=0.6, ax=axes[ix], square=True,
                cbar_kws={"shrink": .5}, xticklabels=2)
    axes[0].set(title=f"Random Cells/Samples")
    axes[1].set(title=f"Ident Cells/Samples")
    # axes[0].text(0, -0.2, f"Correlation (Spearman): {corr_rand_ident}", size=10,
    #              transform=axes[0].transAxes)

save_corr("RandIdent_heatmap_correlation")
plt.close()


boxplot_df1 = pd.DataFrame()
boxplot_df2 = pd.DataFrame()

rand_corr_values = upper(rand_corr)
boxplot_df1["corr_values"] = rand_corr_values
boxplot_df1["labels"] = "Random Cells/Samples"

ident_corr_values = upper(ident_corr)
boxplot_df2["corr_values"] = ident_corr_values
boxplot_df2["labels"] = "Ident Cells/Samples"

df_merged = pd.concat([boxplot_df1, boxplot_df2], ignore_index=True, sort=False, axis=0)

sns.boxplot(df_merged, x="labels", y="corr_values")
plt.title("Difference between identical cell input vs. random cell input")
# plt.figtext(x=0, y=0.05, s=f"Correlation (Spearman): {corr_rand_ident}", size=10)
save_corr("RandIdent_boxplot_correlation")
