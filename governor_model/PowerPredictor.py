from DatasetBuilder import importDataset, pp2B
import torch.nn as nn
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

freqLevels = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000,
              1704000, 1800000, 1908000, 2016000, 2100000, 2208000]

V_RATE_E = 100
## HYPERPARAMETERS WE SHOULD TEST:
N_EPOCHS = 10000
BATCH_SIZE = 30
LEARNING_RATE = 7e-6

## first architecture: seems to be impossible to train, too hard for it to figure out.
# INPUT FEATURES:
#   - partition point 1
#   - partition point 2
#   - bfrequency
#   - bfrequency**2
#   - lfrequency
#   - lfrequency**2
#   = 6

## other possible architectures:
#   nn option 2:
#   - s1 size
#   - s2 size
#   - s3 size
#   - bfrequency
#   - bfrequency**2
#   - lfrequency
#   - lfrequency**2
#   = 7
#
#   nn option 3:
#   -*binary layers s1
#   -*binary layers s2
#   -*binary layers s3
#   - bfrequency
#   - bfrequency**2
#   - lfrequency
#   - lfrequency**2
#   = 25
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
def make_model():
    return nn.Sequential(
        nn.Linear(1, 1),
        nn.Linear(1, 1)
    )


# should not port
def train(n_epochs=5000, batch_size=20, lr=0.0001, verbose=False, graph=False):
    model = make_model()
    X_train, y_train, X_test, y_test = importDataset(train_size=0.8)

    loss_fn = nn.MSELoss()  # mean square error
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.6)

    batch_positions = np.arange(0, len(X_train), batch_size)

    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []
    history_t = []

    for epoch in range(n_epochs):
        if verbose and epoch % V_RATE_E == 0:
            print(f"Epoch {epoch+1}\n-------------------------------")


        model.train() # set model to train mode
        for batch_n, batch_start in enumerate(batch_positions):

            # take a batch
            X_batch = X_train[batch_start:batch_start+batch_size]
            y_batch = y_train[batch_start:batch_start+batch_size]
            # if verbose and epoch % V_RATE_E == 0 and batch_n == 0:
            #     print(y_batch)

            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and epoch % V_RATE_E == 0 and batch_n % 5 == 0:
                loss, current = loss.item(), (batch_n + 1) * batch_size
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(X_train):>5d}]")

        model.eval()  # set model to evaluate mode
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        y_pred_train = model(X_train)
        mse_t = loss_fn(y_pred_train, y_train)
        mse = float(mse)
        mse_t = float(mse_t)

        if verbose and epoch % V_RATE_E == 0:
            print(f"Test mean-squared error: {mse:>8f} \n")

        history.append(mse)
        history_t.append(mse_t)
        # history_t.append(mse_t)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    # load model with best accuracy
    model.load_state_dict(best_weights)
    with open("weights/best_score_pwr.txt", "r+") as f:
        raw_record = f.readline().strip()
        if raw_record == '':
            raw_record = '800000'
        record_mse = float(raw_record)
        if best_mse < record_mse:
            torch.save(best_weights, f"weights/power_weights.weights")
            record_mse = best_mse
            f.truncate(0)
            f.seek(0)
            f.write(f'{best_mse}\n')
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



def build_pwr_predictor():
    model = make_model()
    model.load_state_dict(torch.load(f"weights/power_weights.weights"))
    model.eval()
    return model


# should port
def predict_power(chromosome, model) -> tuple[float]:
    pp1 = chromosome[0].layers
    pp2 = pp1 + chromosome[1].layers
    bfreq = freqLevels[chromosome[0].frequency_level]
    lfreq = freqLevels[chromosome[2].frequency_level]
    pwr = float(model([pp1, pp2, bfreq, lfreq]))
    return pwr


if __name__ == "__main__":
    train(N_EPOCHS, BATCH_SIZE, LEARNING_RATE, verbose=True)
