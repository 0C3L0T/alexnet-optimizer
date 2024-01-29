"""
This is a tester to help us find a way to calculate total latency from
inference. It's just a messy script, so don't take the code quality seriously.
"""
from itertools import accumulate
from measurementAggregator import Aggregator
import numpy as np

LAYERS = 8
GHZ = 1000000
ADJUST = 0.9
TRAINSPLIT = 0.7
a = Aggregator()
a.aggregate("training_data/perf_data_raw.txt", None)

input_lat_diff = []
for n, sample in enumerate(a.split[:int(len(a.split)*0.7)]):
    pp1 = sample[0]["pp1"]
    pp2 = sample[0]["pp2"]
    s1_inference = sample[1]["s1_inference"] if "s1_inference" in sample[1].keys() else 0.0
    s2_inference = sample[1]["s2_inference"] if "s2_inference" in sample[1].keys() else 0.0
    s3_inference = sample[1]["s3_inference"] if "s3_inference" in sample[1].keys() else 0.0
    s1_input = sample[1]["s1_input"] if "s1_input" in sample[1].keys() else 0.0
    s2_input = sample[1]["s2_input"] if "s2_input" in sample[1].keys() else 0.0
    s3_input = sample[1]["s3_input"] if "s3_input" in sample[1].keys() else 0.0

    true_total_lat = sum([s1_inference, s2_inference, s3_inference, s1_input, s2_input, s3_input])
    active_lat = [s1_inference]
    if pp2-pp1 > 0:
        active_lat.append(s2_inference)
    if pp2 < 8:
        # guess = s3_inference
        active_lat.append(s3_inference)

    prd_total_lat = sum(accumulate(active_lat, max))
    if pp2 < 8 and pp2-pp1 == 0:
        prd_total_lat -= max(0, s1_inference-s3_inference)

    input_lat_diff.append(true_total_lat - prd_total_lat)

print("\n".join([f"{x+3}:{y:.1f}" for x,y in filter(lambda x: abs(x[1])>3.0, list(enumerate(input_lat_diff)))]))

input_lat_diff = np.array(input_lat_diff)
avgdiff = np.mean(input_lat_diff)
stddev = np.mean(np.abs(np.array(input_lat_diff)))
print(avgdiff, stddev)

input_lat_diff_test = []
for sample in a.split[int(len(a.split)*0.7):]:
    pp1 = sample[0]["pp1"]
    pp2 = sample[0]["pp2"]
    s1_inference = sample[1]["s1_inference"] if "s1_inference" in sample[1].keys() else 0.0
    s2_inference = sample[1]["s2_inference"] if "s2_inference" in sample[1].keys() else 0.0
    s3_inference = sample[1]["s3_inference"] if "s3_inference" in sample[1].keys() else 0.0
    s1_input = sample[1]["s1_input"] if "s1_input" in sample[1].keys() else 0.0
    s2_input = sample[1]["s2_input"] if "s2_input" in sample[1].keys() else 0.0
    s3_input = sample[1]["s3_input"] if "s3_input" in sample[1].keys() else 0.0

    true_total_lat = sum([s1_inference, s2_inference, s3_inference, s1_input, s2_input, s3_input])
    active_lat = [s1_inference]
    if pp2-pp1 > 0:
        active_lat.append(s2_inference)
    if pp2 < 8:
        # guess = s3_inference
        # if pp2-pp1 == 0:
        #     guess -= ADJUST*(pp2/8)*s3_inference
        active_lat.append(s3_inference)

    prd_total_lat = sum(accumulate(active_lat, max)) #`+ avgdiff
    if pp2 < 8 and pp2-pp1 == 0:
        prd_total_lat -= max(0, s1_inference-s3_inference)
    # prd_total_lat += 0.7*avgdiff

    input_lat_diff_test.append(true_total_lat - prd_total_lat)

input_lat_diff_test = np.array(input_lat_diff_test)
avgdiff_test = np.mean(input_lat_diff_test)
stddev_test = np.mean(np.abs(input_lat_diff_test))
print(avgdiff_test, stddev_test)
