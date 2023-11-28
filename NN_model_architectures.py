import torch
import torch.nn as nn

# --------- model without sigmoid -----------
class BinaryClassificationWithoutSigmoid(nn.Module):
    """Model has no sigmoid activation before output. Outputs logits"""
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(8114)

        self.linear_layer1 = nn.Linear(in_features=8114,
                                       out_features=8114)

        self.linear_layer2 = nn.Linear(in_features=8114,
                                       out_features=1)

        self.relu = nn.ReLU()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x)
        x = self.relu(self.linear_layer1(x))
        x = self.linear_layer2(x)
        return x.squeeze()

# ------ model with sigmoid -----
class BinaryClassificationWithSigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(8114)

        self.linear_layer1 = nn.Linear(in_features=8114,
                                       out_features=8114)

        self.linear_layer2 = nn.Linear(in_features=8114,
                                       out_features=1)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x)
        x = self.relu(self.linear_layer1(x))

        x = self.sigmoid(self.linear_layer2(x))
        return x.squeeze()


# -------- model with dropout ----------
class BinaryClassificationWithDropout(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(8114)

        self.linear_layer1 = nn.Linear(in_features=8114,
                                       out_features=8114)

        self.linear_layer2 = nn.Linear(in_features=8114,
                                       out_features=1)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x)
        x = self.relu(self.linear_layer1(x))
        x = self.dropout(x)
        # output layer
        x = self.sigmoid(self.linear_layer2(x))
        return x.squeeze()

