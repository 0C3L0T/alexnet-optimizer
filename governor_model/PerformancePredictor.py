from DatasetBuilder import importPerformance, pp2B
import torch.nn as nn
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from itertools import accumulate
from GA import Chromosome

freqLevels = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000,
              1704000, 1800000, 1908000, 2016000, 2100000, 2208000]

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
#   - bfrequency int
#
#   S3:
#   - hasL2 bool
#   - hasL3 bool
#   - hasL4 bool
#   - hasL5 bool
#   - hasL6 bool
#   - hasL7 bool
#   - hasL8 bool
#   - lfrequency int
#


# should port
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)

# should not port
def train(stage, n_epochs=50, batch_size=20):
    X_train, y_train, X_test, y_test = importPerformance(stage)

    loss_fn = nn.MSELoss()  # mean square error
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    batch_positions = np.arange(0, len(X_train), batch_size)

    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []

    for epoch in range(n_epochs):
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

            if batch_n % 100 == 0:
                loss, current = loss.item(), (batch_n + 1) * batch_size
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(X_train):>5d}]")

        model.eval()  # set model to evaluate mode
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)

        print(f"Test mean-squared error: {mse:>8f} \n")

        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    # load model with best accuracy
    model.load_state_dict(best_weights)
    torch.save(best_weights, "weights/s{stage}_weights.txt")
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    plt.plot(history)
    plt.show()

# should port
def predict_stage(stage, X):
    model.load_state_dict(torch.load(f"weights/s{stage}_weights.txt"))
    model.eval()

    return float(model(X))

# should port
def predict_performance(chromosome: Chromosome) -> tuple[float]:
    pp1 = chromosome[0].layers
    pp2 = pp1 + chromosome[1].layers
    bfreq = freqLevels[chromosome[0].frequency_level]
    lfreq = freqLevels[chromosome[2].frequency_level]
    X1 = pp2B(pp1, pp2, 1) + [bfreq]
    X2 = pp2B(pp1, pp2, 2) + [bfreq]
    X3 = pp2B(pp1, pp2, 3) + [lfreq]
    inf_lat1 = predict_stage(1, X1)
    inf_lat2 = predict_stage(2, X2)
    inf_lat3 = predict_stage(3, X3)
    total_lat = sum(accumulate([inf_lat1, inf_lat2, inf_lat3], max))
    max_lat = max(inf_lat1, inf_lat2, inf_lat3)
    return total_lat, max_lat
