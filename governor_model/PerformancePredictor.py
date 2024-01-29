from DatasetBuilder import importDataset, pp2B
import torch.nn as nn
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from itertools import accumulate

"""
I consider this document to just kinda be a cobbled together script, not really a
proper piece of code, so forget about good code practices just for this file ok.
"""

freqLevels = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000,
              1704000, 1800000, 1908000, 2016000, 2100000, 2208000]

## HYPERPARAMETERS WE SHOULD TEST:
N_EPOCHS = 20000
BATCH_SIZE = 20
LEARNING_RATE = 2e-4

# INPUT FEATURES:
#
#   S1:
#   - hasL2 bool
#   - hasL3 bool
#   - hasL4 bool
#   - hasL5 bool
#   - hasL6 bool
#   - hasL7 bool
#   - hasL8 bool
#   - bfrequency int
#
#   S2:
#   - hasL2 bool
#   - hasL3 bool
#   - hasL4 bool
#   - hasL5 bool
#   - hasL6 bool
#   - hasL7 bool
#   - hasL8 bool
#   - clock time int
#
#   S3:
#   - hasL2 bool
#   - hasL3 bool
#   - hasL4 bool
#   - hasL5 bool
#   - hasL6 bool
#   - hasL7 bool
#   - hasL8 bool
#   - clock time int
#


def make_model():
    return nn.Sequential(
        nn.Linear(8, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1)
    )


# should port
model = make_model()


# should not port
def train(stage, n_epochs=50, batch_size=20, lr=0.0001, verbose=False, graph=False):
    X_train, y_train, X_test, y_test = importDataset(stage=stage, train_size=0.6)
    print(X_train.dtype)

    loss_fn = nn.MSELoss()  # mean square error
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print(len(X_train), batch_size)
    batch_positions = np.arange(0, len(X_train), batch_size)

    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []
    history_t = []

    for epoch in range(n_epochs):
        if verbose and epoch % 1000 == 0:
            print(f"Epoch {epoch+1}\n-------------------------------")

        model.train() # set model to train mode
        for batch_n, batch_start in enumerate(batch_positions):

            # take a batch
            X_batch = X_train[batch_start:batch_start+batch_size]
            y_batch = y_train[batch_start:batch_start+batch_size]

            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and epoch % 1000 == 0:## and batch_n % 5 == 0:
                loss, current = loss.item(), (batch_n + 1) * batch_size
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(X_train):>5d}]")

        model.eval()  # set model to evaluate mode
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        y_pred_train = model(X_train)
        mse_t = loss_fn(y_pred_train, y_train)
        mse = float(mse)
        mse_t = float(mse_t)

        if verbose and epoch % 1000 == 0:
            print(f"Test mean-squared error: {mse:>8f} \n")

        history.append(mse)
        history_t.append(mse_t)
        # history_t.append(mse_t)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    # load model with best accuracy
    model.load_state_dict(best_weights)
    with open("weights/best_score_perf.txt", "r+") as f:
        raw_record = f.readline().strip().split(' ')
        if raw_record == ['']:
            raw_record = ['800000', '800000', '800000']
        record_mse = list(map(float, raw_record))
        if best_mse < record_mse[stage-1]:
            torch.save(best_weights, f"weights/s{stage}_weights.weights")
            record_mse[stage-1] = best_mse
            f.truncate(0)
            f.seek(0)
            f.write(' '.join(map(str, record_mse)) + '\n')
            print("Record score!")

    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    if graph:
        # plt.ylim(0,30000)
        # plt.xlim(50,n_epochs)
        x = range(len(history))
        plt.plot(x, history, color="tab:blue", label='test')
        plt.plot(x, history_t, color="tab:red", label='train')
        plt.legend()
        plt.legend()
        plt.show()


def eval_models():
    for i in range(1,4):
        *X, y = np.loadtxt(f"training_data/s{i}_perf_data.txt", unpack=True)
        X = torch.tensor(np.array(X).T, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        model.load_state_dict(torch.load(f"weights/s{i}_weights.weights"))
        model.eval()
        y_pred = model(X)
        loss_fn = nn.MSELoss()
        mse = float(loss_fn(y_pred, y))
        print(f"s{i} MSE: {mse:.2f}\tRMSE: {np.sqrt(mse):.2f}")


# should port
def build_perf_predictors():
    models = [make_model() for _ in range(3)]
    for i, m in enumerate(models):
        m.load_state_dict(torch.load(f"weights/s{i+1}_weights.weights"))
        m.eval()
    return models


# # should port
# def predict_stage(stage, X):
#     model.load_state_dict(torch.load(f"weights/s{stage}_weights.weights"))
#     model.eval()
#     return float(model(X))


# should port
def predict_performance(chromosome, s1, s2, s3) -> tuple[float]:
    pp1 = chromosome[0].layers
    pp2 = pp1 + chromosome[1].layers
    bfreq = freqLevels[chromosome[0].frequency_level]
    lfreq = freqLevels[chromosome[2].frequency_level]
    X1 = torch.tensor(pp2B(pp1, pp2, 1) + [bfreq], dtype=torch.float32)
    X2 = torch.tensor(pp2B(pp1, pp2, 2) + [bfreq], dtype=torch.float32)
    X3 = torch.tensor(pp2B(pp1, pp2, 3) + [lfreq], dtype=torch.float32)
    print("in:", X1, X2, X3)
    inf_lat1 = s1(X1)
    inf_lat2 = s2(X2)
    inf_lat3 = s3(X3)
    print("out:", inf_lat1, inf_lat2, inf_lat3)
    active_lat = [inf_lat1]
    if pp2-pp1 > 0:
        active_lat.append(inf_lat2)
    if pp2 < 8:
        active_lat.append(inf_lat3)
    total_lat = sum(accumulate(active_lat, max))
    max_lat = max(inf_lat1, inf_lat2, inf_lat3)

    #special adjustment
    if pp2 < 8 and pp2-pp1 == 0:
        total_lat -= max(0, inf_lat1-inf_lat3)

    return total_lat, max_lat


if __name__ == "__main__":
    train(1, N_EPOCHS, BATCH_SIZE, LEARNING_RATE, verbose=True)
    train(2, N_EPOCHS, BATCH_SIZE, LEARNING_RATE, verbose=True)
    train(3, N_EPOCHS, BATCH_SIZE, LEARNING_RATE, verbose=True)
