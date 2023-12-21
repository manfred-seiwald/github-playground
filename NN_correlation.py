import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. comment from Manfred
from NN_utils import save_corr, upper

# get attributes
no_dropout_df = pd.read_csv("data/no_dropout_attr.csv")
dropout20_df = pd.read_csv("data/dropout20_attr.csv")
dropout40_df = pd.read_csv("data/dropout40_attr.csv")
dropout50_df = pd.read_csv("data/dropout50_attr.csv")
dropout60_df = pd.read_csv("data/dropout60_attr.csv")
dropout80_df = pd.read_csv("data/dropout80_attr.csv")
# Für die Horde !!!!!!!!!!!!!!!!!!
# transpose dataframe
no_dropout_df = no_dropout_df.T
dropout20_df = dropout20_df.T
dropout40_df = dropout40_df.T
dropout50_df = dropout50_df.T
dropout60_df = dropout60_df.T
dropout80_df = dropout80_df.T
# rename columns to differ model
no_dropout_df.columns = [f"{col_name}.0" for col_name in no_dropout_df.columns]
dropout20_df.columns = [f"{col_name}.2" for col_name in dropout20_df.columns]
dropout40_df.columns = [f"{col_name}.4" for col_name in dropout40_df.columns]
dropout50_df.columns = [f"{col_name}.5" for col_name in dropout50_df.columns]
dropout60_df.columns = [f"{col_name}.6" for col_name in dropout60_df.columns]
dropout80_df.columns = [f"{col_name}.8" for col_name in dropout80_df.columns]
# calculate correlation
corr_no_dropout = no_dropout_df.corr()
corr_dropout20 = dropout20_df.corr()
corr_dropout40 = dropout40_df.corr()
corr_dropout50 = dropout50_df.corr()
corr_dropout60 = dropout60_df.corr()
corr_dropout80 = dropout80_df.corr()

corr_list = [corr_no_dropout, corr_dropout20, corr_dropout40, corr_dropout50, corr_dropout60, corr_dropout80]
drop_out_values = [0, 20, 40, 50, 60, 80]

# -------------- plots -----------
center = 0.5
vmax = 1
vmin = 0.5

def plot_heatmap():
    # -------------- single heatmap ------
    for item, drop in zip(corr_list,drop_out_values):
        fig = plt.figure()
        sns.heatmap(item, cmap="coolwarm",center=center, vmax=vmax, vmin=vmin, square=True,
                    cbar_kws={"shrink": .5}, xticklabels=2)
        fig.subplots_adjust(bottom=0.2)
        plt.title(f"Correlation of feature importance with dropout {drop}%")
        plt.xlabel("Model")
        plt.ylabel("Model")
        save_corr(f"dropout{drop}_heatmap")


def cluster_map():
    # ------- clustermap ----------
    for item, drop in zip(corr_list,drop_out_values):
        g = sns.clustermap(corr_no_dropout, cmap="coolwarm", center=center, vmax=vmax, vmin=vmin, figsize=(8, 5))
        g.ax_col_dendrogram.set_visible(False)
        g.cax.set_visible(False)
        # new comment from Manfred
        g.ax_heatmap.set_title(f"no_dropout")
        save_corr(f"no_dropout_clustermap") # interruption

# -------- subplots of heatmaps ------------

plot_heatmap()

cluster_map()
#comment from Anna and Selina!

fig, axs = plt.subplots(2, 3)
fig.tight_layout()
sns.heatmap(corr_no_dropout, cmap="coolwarm",center=center, vmax=vmax, vmin=vmin, square=True,
            cbar_kws={"shrink": .5}, xticklabels=2, ax=axs[0, 0])
axs[0, 0].set_title("no_dropout")
sns.heatmap(corr_dropout20, cmap="coolwarm",center=center, vmax=vmax, vmin=vmin, square=True,
            cbar_kws={"shrink": .5}, xticklabels=2, ax=axs[0, 1])
axs[0, 1].set_title("dropout_20")
sns.heatmap(corr_dropout40, cmap="coolwarm",center=center, vmax=vmax, vmin=vmin, square=True,
            cbar_kws={"shrink": .5}, xticklabels=2, ax=axs[0, 2])
axs[0, 2].set_title("dropout_40")

sns.heatmap(corr_dropout50, cmap="coolwarm",center=center, vmax=vmax, vmin=vmin, square=True,
            cbar_kws={"shrink": .5}, xticklabels=2, ax=axs[1, 0])
axs[1, 0].set_title("dropout_50")

sns.heatmap(corr_dropout60, cmap="coolwarm",center=center, vmax=vmax, vmin=vmin, square=True,
            cbar_kws={"shrink": .5}, xticklabels=2, ax=axs[1, 1])
axs[1, 1].set_title("dropout_60")

sns.heatmap(corr_dropout80, cmap="coolwarm",center=center, vmax=vmax, vmin=vmin, square=True,
            cbar_kws={"shrink": .5}, xticklabels=2, ax=axs[1, 2])
axs[1, 2].set_title("dropout_80")

for i, row in enumerate(axs):
    for j, cell in enumerate(row):
        if i == len(axs) - 1:
            cell.set_xlabel("Models")
        if j == 0:
            cell.set_ylabel("Models")



# --------- ----------
fig, axs = plt.subplots(1, 2)
fig.tight_layout()
sns.heatmap(corr_dropout60, cmap="coolwarm",center=center, vmax=vmax, vmin=vmin, square=True,
            cbar_kws={"shrink": .5}, xticklabels=2, ax=axs[0])
axs[0].set_xlabel("Model")
axs[0].set_ylabel("Model")
axs[0].set_title("dropout_60")
sns.heatmap(corr_dropout80, cmap="coolwarm",center=center, vmax=vmax, vmin=vmin, square=True,
            cbar_kws={"shrink": .5}, xticklabels=2, ax=axs[1])
axs[1].set_xlabel("Model")
axs[1].set_ylabel("Model")
axs[1].set_title("dropout_80")




# --------------- correlation bewtween all models ---------

df = pd.concat([no_dropout_df, dropout20_df], axis=1)
corr = df.corr()




# ------- violin plot --------

violin_no = pd.DataFrame()
violin_20 = pd.DataFrame()
violin_40 = pd.DataFrame()
violin_50 = pd.DataFrame()
violin_60 = pd.DataFrame()
violin_80 = pd.DataFrame()

# comment by Daniel Katzlberger


no_dropout_values = upper(corr_no_dropout)
violin_no["corr_values"] = no_dropout_values
violin_no["dropout_values"] = 0
violin_no["labels"] = "no_dropout"

dropout20_values = upper(corr_dropout20)
violin_20["corr_values"] = dropout20_values
violin_20["dropout_values"] = 20
violin_20["labels"] = "dropout20"

dropout40_values = upper(corr_dropout40)
violin_40["corr_values"] = dropout40_values
violin_40["dropout_values"] = 40
violin_40["labels"] = "dropout40"

dropout50_values = upper(corr_dropout50)
violin_50["corr_values"] = dropout50_values
violin_50["dropout_values"] = 50
violin_50["labels"] = "dropout50"

dropout60_values = upper(corr_dropout60)
violin_60["corr_values"] = dropout60_values
violin_60["dropout_values"] = 60
violin_60["labels"] = "dropout60"

dropout80_values = upper(corr_dropout80)
violin_80["corr_values"] = dropout80_values
violin_80["dropout_values"] = 80
violin_80["labels"] = "dropout80"

violin_merge = pd.concat([violin_no, violin_20, violin_40, violin_50, violin_60, violin_80], ignore_index=True, sort=False, axis=0)

sns.violinplot(violin_merge, x="labels", y="corr_values")
plt.title("Comparison of drop_out rates")
save_corr("violinplot")


