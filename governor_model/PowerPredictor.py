from DatasetBuilder import importDataset, pp2B
import torch.nn as nn
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

freqLevels = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000,
              1704000, 1800000, 1908000, 2016000, 2100000, 2208000]

## HYPERPARAMETERS WE SHOULD TEST:
N_EPOCHS = 50
BATCH_SIZE = 20
LEARNING_RATE = 1e-4

# INPUT FEATURES:
#
#   - partition point 1
#   - partition point 2
#   - bfrequency
#   - bfrequency**2
#   - lfrequency
#   - lfrequency**2

## other possible architectures:
#   nn option 2:
#   - s1 size
#   - s2 size
#   - s3 size
#   - bfrequency
#   - bfrequency**2
#   - lfrequency
#   - lfrequency**2
#
#   nn option 3:
#   -*binary layers s1
#   -*binary layers s2
#   -*binary layers s3
#   - bfrequency
#   - bfrequency**2
#   - lfrequency
#   - lfrequency**2
#
#   m1:
#   like perf s1 but with bfrequency**2
#   m2:
#   like perf s2 but with bfrequency**2
#   m3:
#   like perf s3 but with lfrequency**2
#
#   nn option 4:
#   m1, m2, m3 simulataneously to 3 outputs -> Activation() -> linear(3,1)


# should port
model = nn.Sequential(
    nn.Linear(6, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)


# should not port
def train(n_epochs=5000, batch_size=20, lr=0.0001):
    X_train, y_train, X_test, y_test = importDataset()

    loss_fn = nn.MSELoss()  # mean square error
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

            if batch_n % 5 == 0:
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
    torch.save(best_weights, "weights/power_weights.txt")
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    plt.plot(history)
    plt.show()


# should port
def predict_power(chromosome) -> tuple[float]:
    pp1 = chromosome[0].layers
    pp2 = pp1 + chromosome[1].layers
    bfreq = freqLevels[chromosome[0].frequency_level]
    lfreq = freqLevels[chromosome[2].frequency_level]
    model.load_state_dict(torch.load(f"weights/power_weights.txt"))
    model.eval()
    pwr = float(model([pp1, pp2, bfreq, lfreq]))
    return pwr


if __name__ == "__main__":
    train(N_EPOCHS, BATCH_SIZE, LEARNING_RATE)
