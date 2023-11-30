import torch
from torch.utils.data import DataLoader, random_split


from NN_datasets import ReactomeDataset
from NN_model_architectures import BinaryClassificationWithSigmoid, BinaryClassificationWithDropout, BinaryClassificationWithDropoutSmall
from NN_utils import logits_learner, probability_learner

# --------- Loading and Splitting Dataset ----------
dataset = ReactomeDataset()

train_size = int(0.8 * len(dataset))  # 80% of the dataset
test_size = len(dataset) - train_size

trainset, testset = random_split(dataset, [train_size, test_size])


# --------- load the data into Dataloader --------
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)  # shuffling does not make sense as we are not learning


# ----------------- train/test loop ---------------
for i in range(7):
    print(i + 1)
    #model = BinaryClassificationWithDropout(dropout=0.6)
    model = BinaryClassificationWithDropoutSmall(dropout=0.6)
    probability_learner(model, trainloader, testloader, 50, 0.01, "BinaryClassificationWithDropout60", printprogress=True)
