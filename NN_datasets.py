import torch
from torch.utils.data import Dataset
import pandas as pd

class ReactomeDataset(Dataset):
    """Reactome Dataset"""
    def __init__(self):

        df = pd.read_csv("data/Reactome_TCR.csv")

        # Setting target
        self.label = "TCR"

        # Save labels and features
        self.features = df.drop(self.label, axis=1)
        self.labels = df[self.label]

        # getting length of the df
        self.n_samples = df.shape[0]

        # getting columns
        # self.feature_names = self.features.columns

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        #convert tensor to list
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.features.iloc[idx].values, self.labels[idx]]
