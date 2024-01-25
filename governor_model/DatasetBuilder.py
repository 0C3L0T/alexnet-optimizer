from measurementAggregator import Aggregator

LAYERS = 8

def pp2B(pp1, pp2, stage):
    if stage == 1:
        return [1]*(pp1-1)+[0]*(LAYERS-pp1)
    if stage == 2:
        return [0]*(pp1-1)+[1]*(pp2-pp1)+[0]*(LAYERS-pp2)
    if stage == 3:
        return [0]*(pp2-1)+[1]*(LAYERS-pp2)

def exportPerformance():
    a = Aggregator()
    # a.aggregate("training_data/perfData.txt", None)
    a.aggregate("../data_collection/test_results/single_components/together/1/adb_output.txt", None)
    print(a.split[0])

    s1_data = []
    s2_data = []
    s3_data = []
    for sample in a.split:
        pp1 = sample[0]["pp1"]
        pp2 = sample[0]["pp2"]
        bfreq = sample[0]["pp2"]
        lfreq = sample[0]["pp2"]
        s1_inference = sample[1]["s1_inference"] if "s1_inference" in sample[1].keys() else 0.0
        s2_inference = sample[1]["s2_inference"] if "s2_inference" in sample[1].keys() else 0.0
        s3_inference = sample[1]["s3_inference"] if "s3_inference" in sample[1].keys() else 0.0
        s1_data.append(pp2B(pp1,pp2,1) + [bfreq, s1_inference])
        s2_data.append(pp2B(pp1,pp2,2) + [bfreq, s2_inference])
        s3_data.append(pp2B(pp1,pp2,3) + [lfreq, s3_inference])

    with open("training_data/s1_perf_data.txt") as f:
        f.write("\n".join([" ".join(sample) for sample in s1_data]))
    with open("training_data/s2_perf_data.txt") as f:
        f.write("\n".join([" ".join(sample) for sample in s2_data]))
    with open("training_data/s3_perf_data.txt") as f:
        f.write("\n".join([" ".join(sample) for sample in s3_data]))

def exportPower():
    a = Aggregator()
    # a.aggregate("training_data/pwrData.txt", None)
    a.aggregate("../data_collection/test_results/single_components/together/1/adb_output.txt", None)
    print(a.split[0])

    data = []
    for sample in a.split:
        pp1 = sample[0]["pp1"]
        pp2 = sample[0]["pp2"]
        bfreq = sample[0]["pp2"]
        lfreq = sample[0]["pp2"]
        try:
            power = sample[1]["avgPower"]
        except KeyError:
            print("This measurement set is missing power data, can't build")
            return
        data.append([pp1,pp2,bfreq,lfreq,power])

    with open("training_data/s1_perf_data.txt") as f:
        f.write("\n".join([" ".join(sample) for sample in data]))


def importPerformance(stage:int, train_size=0.7):
    from sklearn.model_selection import train_test_split
    import torch
    import numpy as np

    # train-test split for model evaluation
    f = open(f"training_data/s{stage}_perf_data.txt")
    *X, y = np.loadtxt(f"training_data/s{stage}_perf_data.txt", unpack=True)
    X = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True)

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("Y_train:", y_train.shape)
    print("Y_test:", y_test.shape)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    return X_train, y_train, X_test, y_test
