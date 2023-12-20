import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryROC, BinaryConfusionMatrix, BinaryRecall, BinaryPrecision, BinaryF1Score

from captum.attr import IntegratedGradients

import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
import numpy as np

from NN_datasets import ReactomeDataset

# ------------- accuracy ----------------------
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))
    return acc

# ----------- training -------------
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device = "cuda" if torch.cuda.is_available() else "cpu"):
    """Trains the model on a DataLoader."""
    train_loss, train_acc = 0, 0

    # put model into training mode
    model.train()

    # Add loop to loop through the training batches
    for batch, (train_inputs, train_labels) in enumerate(data_loader):

        # getting inputs and labels to right device
        train_inputs.to(device)
        train_labels.to(device)

        # forward pass
        train_logits = model(train_inputs.float())#.squeeze() for right shape of tensor model might return squeezed tensor; inputs maybe need .float()
        train_pred = torch.round(torch.sigmoid(train_logits)) # model does output logits

        # calculating loss and acc
        loss = criterion(train_logits, train_labels.float())
        train_loss += loss
        acc = torch.eq(train_labels, train_pred).sum().item() / float(train_labels.size(0))
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc

#----------------- test ----------------------

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              device = "cuda" if torch.cuda.is_available() else "cpu"):
    """Testing the Model on a DataLoader"""
    test_loss, test_acc = 0, 0

    # put model in eval mode
    model.eval()
    # inference mode
    with torch.inference_mode():
        for test_inputs, test_labels in data_loader:

            # getting them to devie
            test_inputs = test_inputs.to(device)
            test_inputs = test_labels.to(device)

            # forward pass
            test_logits = model(test_inputs.float())
            test_pred = torch.round(torch.sigmoid(test_logits))

            # calculating loss and acc
            loss = criterion(test_logits, test_labels.float())
            test_loss += loss
            acc = torch.eq(test_labels, test_pred).sum().item() / float(test_labels.size(0))
            test_acc += acc

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        return test_loss, test_acc

# -------------- saving model -----------

def save_model(name, model):
    MODEL_PATH = Path("trained_models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

# --------- save figure ------------

def save_metric_figure(name):
    FIGURE_PATH = Path("metric_figures")
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)

    FIGURE_NAME = name
    FIGURE_SAVE_PATH = FIGURE_PATH / FIGURE_NAME

    plt.savefig(FIGURE_SAVE_PATH)

# ------------ learner --------------
def logits_learner(model, trainloader, testloader, epochs, lr, save_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model
    criterion = nn.BCEWithLogitsLoss()
    lr = lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # metric collection
    bconfmat = BinaryConfusionMatrix()
    broc = BinaryROC()

    collection = torchmetrics.MetricCollection(
        BinaryAccuracy(), BinaryRecall(),
        BinaryPrecision(), BinaryAUROC(thresholds=None),
        BinaryF1Score(), bconfmat, broc)

    test_tracker = torchmetrics.wrappers.MetricTracker(collection)
    train_tracker = torchmetrics.wrappers.MetricTracker(collection)

    NUM_EPOCHS = epochs
    curr_best_auc = 0.0  # initializing best metric score
    patience = 10  # NUM_EPOCHS before termination of loop after no increase in metric
    print(f"Logits learner start")
    print(f"\nEpoch:"
          f"\tTrain Loss:"
          f"\tTest Loss:"
          f"\tTrain AUC:"
          f"\tTest AUC:"
          f"\tTrain Acc:"
          f"\tTest Acc:")

    for epoch in range(NUM_EPOCHS):
        # training the model
        train_loss = 0.0
        test_loss = 0.0

        train_tracker.increment()
        test_tracker.increment()

        for batch, (train_inputs, train_labels) in enumerate(trainloader):
            # getting inputs and labels to right device
            train_inputs.to(device)
            train_labels.to(device)
            model.to(device)
            # start training
            model.train()
            train_logits = model(train_inputs.float())  # .squeeze() for right shape of tensor
            # train_pred = torch.round(torch.sigmoid(logits))# maybe find a better way to set the threshold than round

            # calculating loss
            loss = criterion(train_logits, train_labels.float())
            train_loss += loss

            # update metrics in tracker
            train_tracker(train_logits, train_labels)

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # train_loss per epoch
        train_loss /= len(trainloader)

        # compute train_tracker
        train_all_results = train_tracker.compute_all()

        train_epoch_auc = train_all_results[epoch]["BinaryAUROC"]
        train_epoch_acc = train_all_results[epoch]["BinaryAccuracy"]

        # Testing the model
        model.eval()
        with torch.inference_mode():

            for test_inputs, test_labels in testloader:
                # getting on the right device
                test_inputs.to(device)
                test_labels.to(device)
                # start testing
                test_logits = model(test_inputs.float())
                # test_pred = torch.round(torch.sigmoid(test_logits))

                # calculating loss
                loss = criterion(test_logits, test_labels.float())
                test_loss += loss

                # update metrics in tracker
                test_tracker.update(test_logits, test_labels)

            # test_loss per epoch
            test_loss /= len(testloader)

        # compute tracker
        test_all_results = test_tracker.compute_all()

        test_epoch_auc = float(test_all_results[epoch]["BinaryAUROC"])
        test_epoch_acc = float(test_all_results[epoch]["BinaryAccuracy"])

        # print metrics per epoch
        print(f"{epoch + 1}"
              f"\t\t{train_loss:.4f}"
              f"\t\t{test_loss:.4f}"
              f"\t\t{train_epoch_auc:.4f}"
              f"\t\t{test_epoch_auc:.4f}"
              f"\t\t{train_epoch_acc:.4f}"
              f"\t\t{test_epoch_acc:.4f}")

        if test_epoch_auc >= curr_best_auc:
            curr_best_auc = test_epoch_auc
            epoch_with_best_auc = (epoch + 1)
            save_model(save_name, model)

            # plotting testing steps
            fig = plt.figure(layout="constrained")
            ax1 = plt.subplot(2, 2, 1)
            ax2 = plt.subplot(2, 2, 2)
            ax3 = plt.subplot(2, 2, (3, 4))

            bconfmat.plot(val=test_all_results[-1]['BinaryConfusionMatrix'], ax=ax1)
            ax1.set_title("Testing-ConfusionMatrix")

            broc.plot(curve=test_all_results[-1]["BinaryROC"], score=True, ax=ax2)
            ax2.set_title("Testing-ROC")

            test_scalar_results = [
                {k: v for k, v in ar.items() if isinstance(v, torch.Tensor) and v.numel() == 1} for ar in
                test_all_results
            ]
            train_tracker.plot(val=test_scalar_results, ax=ax3)

            for line in ax3.lines:
                line.set_marker(".")
            ax3.legend(loc='lower right', ncols=2, fontsize="small")

            save_metric_figure(save_name)
            plt.close()

            patience = 10
        else:
            patience -= 1

        if patience == 0 or (epoch + 1) == NUM_EPOCHS:
            print(
                f"Logits learner terminated after Epoch: {epoch + 1}/{NUM_EPOCHS}, best recorded Test-AUC: {curr_best_auc:.4f} in Epoch: {epoch_with_best_auc}")
            break
    # reset metric trackers
    train_tracker.reset_all()
    test_tracker.reset_all()

    # rename model and metric figure

    model_path = "trained_models/"
    model_save_path = model_path + save_name

    figure_path = "metric_figures/"
    figure_save_path = figure_path + save_name + ".png"

    os.rename(model_save_path, model_path + save_name + "_AUC:" + str(round(curr_best_auc, 4)) + ".pt")
    os.rename(figure_save_path, figure_path + "Metrics_" + save_name + "_AUC:" + str(round(curr_best_auc, 4)) + ".png")

# ------- probability learner -------
def probability_learner(model, trainloader, testloader, epochs, lr, save_name, printprogress = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model
    criterion = nn.BCELoss()
    lr = lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # metric collection
    bconfmat = BinaryConfusionMatrix()
    broc = BinaryROC()

    collection = torchmetrics.MetricCollection(
        BinaryAccuracy(), BinaryRecall(),
        BinaryPrecision(), BinaryAUROC(thresholds=None),
        BinaryF1Score(), bconfmat, broc)

    test_tracker = torchmetrics.wrappers.MetricTracker(collection)
    train_tracker = torchmetrics.wrappers.MetricTracker(collection)

    NUM_EPOCHS = epochs
    curr_best_auc = 0.0  # initializing best metric score
    patience = 10  # NUM_EPOCHS before termination of loop after no increase in metric
    print(f"\nProbability learner start:")
    if printprogress == True:
        print(f"Epoch:"
              f"\tTrain Loss:"
              f"\tTest Loss:"
              f"\tTrain AUC:"
              f"\tTest AUC:"
              f"\tTrain Acc:"
              f"\tTest Acc:")
    else:
        pass

    for epoch in range(NUM_EPOCHS):
        # training the model
        train_loss = 0.0
        test_loss = 0.0

        train_tracker.increment()
        test_tracker.increment()

        for batch, (train_inputs, train_labels) in enumerate(trainloader):
            # getting inputs and labels to right device
            train_inputs = train_inputs.to(device)
            train_labels = train_labels.to(device)
            model.to(device)
            # start training
            model.train()
            train_prob = model(train_inputs.float())  # .squeeze() for right shape of tensor, model might return squeezed tensor;
            # train_pred = torch.round(train_prob)

            # calculating loss
            loss = criterion(train_prob, train_labels.float())
            train_loss += loss

            # update metrics in tracker
            train_tracker(train_prob.to('cpu'), train_labels.to('cpu'))

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # train_loss per epoch
        train_loss /= len(trainloader)

        # compute train_tracker
        train_all_results = train_tracker.compute_all()

        train_epoch_auc = train_all_results[epoch]["BinaryAUROC"]
        train_epoch_acc = train_all_results[epoch]["BinaryAccuracy"]

        # Testing the model
        model.eval()
        with torch.inference_mode():

            for test_inputs, test_labels in testloader:
                # getting on the right device
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)
                # start testing
                test_prob = model(test_inputs.float())
                # test_pred = torch.round(test_prob)

                # calculating loss
                loss = criterion(test_prob, test_labels.float())
                test_loss += loss

                # update metrics in tracker
                test_tracker.update(test_prob.to('cpu'), test_labels.to('cpu'))

            # test_loss per epoch
            test_loss /= len(testloader)

        # compute tracker
        test_all_results = test_tracker.compute_all()

        test_epoch_auc = float(test_all_results[epoch]["BinaryAUROC"])
        test_epoch_acc = float(test_all_results[epoch]["BinaryAccuracy"])

        # print metrics per epoch
        if printprogress == True:
            print(f"{epoch + 1}"
                  f"\t\t{train_loss:.4f}"
                  f"\t\t{test_loss:.4f}"
                  f"\t\t{train_epoch_auc:.4f}"
                  f"\t\t{test_epoch_auc:.4f}"
                  f"\t\t{train_epoch_acc:.4f}"
                  f"\t\t{test_epoch_acc:.4f}")
        else:
            pass

        if test_epoch_auc >= curr_best_auc:
            curr_best_auc = test_epoch_auc
            epoch_with_best_auc = (epoch + 1)
            save_model(save_name, model)

            # plotting testing steps
            fig = plt.figure(layout="constrained")
            ax1 = plt.subplot(2, 2, 1)
            ax2 = plt.subplot(2, 2, 2)
            ax3 = plt.subplot(2, 2, (3, 4))

            bconfmat.plot(val=test_all_results[-1]['BinaryConfusionMatrix'], ax=ax1)
            ax1.set_title("Testing-ConfusionMatrix")

            broc.plot(curve=test_all_results[-1]["BinaryROC"], score=True, ax=ax2)
            ax2.set_title("Testing-ROC")

            test_scalar_results = [
                {k: v for k, v in ar.items() if isinstance(v, torch.Tensor) and v.numel() == 1} for ar in
                test_all_results
            ]
            train_tracker.plot(val=test_scalar_results, ax=ax3)

            for line in ax3.lines:
                line.set_marker(".")
            ax3.legend(loc='lower right', ncols=2, fontsize="small")

            save_metric_figure(save_name)
            plt.close()

            patience = 10
        else:
            patience -= 1

        if patience == 0 or (epoch + 1) == NUM_EPOCHS:
            print(
                f"Probability learner terminated after Epoch: {epoch + 1}/{NUM_EPOCHS}, best recorded Test-AUC: {curr_best_auc:.4f} in Epoch: {epoch_with_best_auc}")
            break
    # reset metric trackers
    train_tracker.reset_all()
    test_tracker.reset_all()

    # rename model and metric figure

    model_path = "trained_models/"
    model_save_path = model_path + save_name

    figure_path = "metric_figures/"
    figure_save_path = figure_path + save_name + ".png"

    os.rename(model_save_path, model_path + save_name + "_AUC:" + str(round(curr_best_auc, 4)) + ".pt")
    os.rename(figure_save_path, figure_path + "Metrics_" + save_name + "_AUC:" + str(round(curr_best_auc, 4)) + ".png")

# ------------------ save correlation ------------------------

def save_corr(name):
    FIGURE_PATH = Path("corr_figures")
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)

    FIGURE_NAME = name
    FIGURE_SAVE_PATH = FIGURE_PATH / FIGURE_NAME

    plt.savefig(FIGURE_SAVE_PATH)

# ------------------------- getting upper part of correlation ---------

def upper(df):
    try:
        assert(type(df) is np.ndarray)
    except:
        if type(df) is pd.DataFrame:
            df = df.values
        else:
            raise TypeError('Must be np.ndarray or pd.DataFrame')
    mask = np.triu_indices(df.shape[0], k=1)
    return df[mask]

# ---------------------- ig calculation --------------

def ig_calculator(directory, model, input_tensor):
    dataset = ReactomeDataset()
    results = pd.DataFrame()
    directory = directory
    count = 0
    print(f"Using {model}.")
    for file in os.listdir(directory):
        count += 1
        print(f"{count}/{len(os.listdir(directory))}")

        path_to_file = f"{directory}/{file}"
        model = model
        model.load_state_dict(torch.load(f=path_to_file))

        ig = IntegratedGradients(model)
        attr, delta = ig.attribute(input_tensor, return_convergence_delta=True)
        attr = attr.detach().numpy()

        mean_attr = np.mean(attr, axis=0)
        mean_attr_df = pd.DataFrame(data=[mean_attr],
                                    columns=dataset.features.columns)

        results = pd.concat([mean_attr_df, results], ignore_index=True)

    return results

# ------------- calculate correlation ------------------

def do_corr(csv_file, extension):
    attr_df = pd.read_csv(csv_file)
    attr_df = attr_df.T
    attr_df.columns = [f"{col_name}.{extension}" for col_name in attr_df.columns]
    corr_attr_df = attr_df.corr()

    return corr_attr_df






