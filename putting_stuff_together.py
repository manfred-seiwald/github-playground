import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --------------- correlation bewtween all models ---------

df = pd.concat([no_dropout_df, dropout20_df, dropout40_df, dropout50_df, dropout60_df, dropout80_df], axis=1)
corr = df.corr()


fig = plt.figure()
plt.title("Correlation between model types")
c = sns.heatmap(corr, cmap="coolwarm", center=center, vmax=vmax, vmin=vmin, cbar_kws={"shrink": .5})
plt.gcf()
c.set_xticks(np.arange(0, 245, 35))
c.set_yticks(np.arange(0, 245, 35))




g = sns.clustermap(corr, cmap="coolwarm", center=0.0, vmax=1, vmin=0.0, figsize=(8, 5))
g.ax_col_dendrogram.set_visible(False)
g.cax.set_visible(False)
g.ax_heatmap.set_title("dropout_80")




