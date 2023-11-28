import pandas as pd
import numpy as np

import torch

from NN_model_architectures import BinaryClassificationWithSigmoid, BinaryClassificationWithDropout

from NN_utils import ig_calculator

# --------------- data ---------------
test_df = pd.read_csv("data/ig_features.csv")
test_features = np.array(test_df)
test_features_tensor = torch.from_numpy(test_features).type(torch.FloatTensor)  # making tensor
test_features_tensor.requires_grad_()

# ------------- calculating mean attributes ------------

attr_df = ig_calculator("trained_models/drop_out60", BinaryClassificationWithDropout(dropout=0.6), test_features_tensor)
attr_df.to_csv("data/dropout60_attr.csv", index=False)
















