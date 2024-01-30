from measurementTools import *
from measurementAggregator import Aggregator
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import re

# # a = Aggregator()
# # a.aggregate("test_results/single_components/adb_output.txt",
# #             "test_results/single_components/power_output.txt",
# #             append=True)
# # a.aggregate("")

# # processOutputs = [f"test_results/single_components/big/{i}/adb_output.txt" for i in range(1,4)] + [f"test_results/single_components/gpu/{i}/adb_output.txt" for i in range(1,3)]
# # powerOutputs = [f"test_results/single_components/big/{i}/power_output.txt" for i in range(1,4)] + [f"test_results/single_components/gpu/{i}/power_output.txt" for i in range(1,3)]
# # a.bulkAggregate(processOutputs, powerOutputs)
# a = Aggregator()
# a.aggregate("test_results/single_components/together/3/adb_output.txt",
#             "test_results/single_components/together/3/power_output.txt")

# single = a.split
# # print(single)
# del a


# b = Aggregator()
# b.aggregate("test_results/order/adb_output.txt",
#             "test_results/order/power_output.txt")

# multi = b.split
# del b

# ### new plot ##################################################################

# littleplotLabels, littlePlotVals = formatToPLT(single, [("Big Frequency", 0), ("GPU On", 0)], ["fps", "avgPower", "Little Frequency"])
# littlePlotVals = np.array(littlePlotVals)
# ##### vals are: freq, pwr, fps #####

# fig, ax1 = plt.subplots()
# littleFreq = littlePlotVals[0]/1e6
# littlePwr = littlePlotVals[2]/1000
# littleFps = littlePlotVals[1]

# color = 'tab:red'
# ax1.set_xlabel('Core clock (GHz)')
# ax1.set_ylabel('FPS', color=color)  # we already handled the x-label with ax1
# ax1.plot(littleFreq, littleFps, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('Power (W)', color=color)
# ax2.plot(littleFreq, littlePwr, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# # fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.title("Little cluster power and performance")
# plt.show()

# ### new plot ##################################################################


# littlefpspw = littleFps/(littlePwr)#-2)
# # print(littleFps)
# # print(littlePwr)
# # print(littlefpspw)

# plt.xlabel('Core clock (GHz)')
# plt.ylabel('FPS/Watt')
# plt.plot(littleFreq, littlefpspw)
# plt.title("Little cluster efficiency")

# plt.show()

# ### new plot ##################################################################

# bigplotLabels, bigPlotVals = formatToPLT(single, [("Little Frequency", 0), ("GPU On", 0)], ["fps", "avgPower", "Big Frequency"])
# bigPlotVals = np.array(bigPlotVals)

# bigFreq = bigPlotVals[0]/1e6
# bigPwr = bigPlotVals[2]/1000
# bigFps = bigPlotVals[1]

# # print((bigFps[10]-bigFps[0])/(bigFreq[10]-bigFreq[0]))
# # print((littleFps[8]-littleFps[0])/(littleFreq[8]-littleFreq[0]))
# bigfpspw = bigFps/(bigPwr)#-2)

# plt.xlabel('Core clock (GHz)')
# plt.ylabel('FPS/Watt')
# plt.plot(littleFreq, littlefpspw, label="Little cluster")
# plt.plot(bigFreq, bigfpspw, label="Big cluster")
# plt.title("Big and little cluster efficiency ")
# plt.legend()

# plt.show()

# ### new plot ##################################################################

# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel('Core clock (GHz)')
# ax1.set_ylabel('FPS', color=color)  # we already handled the x-label with ax1
# ax1.plot(bigFreq, bigFps, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('Power (W)', color=color)
# ax2.plot(bigFreq, bigPwr, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# # fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.title("Big cluster power and performance")
# plt.show()

# ### new plot ##################################################################

# gpuPlotLabels, gpuPlotVals = formatToPLT(single, [("GPU On", 1)], ["fps", "avgPower", "Big Frequency", "s1_input"])
# gpuPlotVals = np.array(gpuPlotVals)

# bigfreqgpu = gpuPlotVals[0]/1e6
# s1inp_time = gpuPlotVals[1]
# fpsgpu = gpuPlotVals[2]
# pwrgpu = gpuPlotVals[3]/1000

# fig, ax1 = plt.subplots()

# # color = 'tab:red'
# ax1.set_xlabel('Core clock of big cluster (GHz)')
# # ax1.set_ylabel('FPS', color=color)  # we already handled the x-label with ax1
# ax1.set_ylabel('input time (seconds)')
# # ax1.plot(bigfreqgpu, fpsgpu, color=color)
# ax1.bar(bigfreqgpu, s1inp_time, width=0.08, color="tab:gray")
# ax1.tick_params(axis='y', labelcolor=color)

# # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# # color = 'tab:blue'
# # ax2.set_ylabel('Power (W)', color=color)
# # ax2.plot(bigfreqgpu, pwrgpu, color=color)
# # ax2.tick_params(axis='y', labelcolor=color)

# # fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.title("GPU only, modulating big frequency")
# plt.show()

# ### new plot ##################################################################

# print(multi)
orderLabels, orderVals1 = formatToPLT(multi, [], ["order", "fps", "avgPower", "s1_inference", "s1_input", "s2_inference", "s2_input"])
orderVals = np.round(np.array(orderVals1[1:]), 0)
# print(orderLabels) ['order', 's1_input', 's1_inference', 's2_input', 's2_inference', 'fps', 'avgPower']

orders = orderVals1[0]
trunc_orders = [order[:3] for order in orders]
di = {ord('B'): 'Big', ord('G'): 'GPU', ord('L'): 'Little'}
expanded_orders = [order.translate(di) for order in trunc_orders]

# print(orderVals[1])
# print(orderVals[2])
s1_total = orderVals[0] + orderVals[1]
s2_total = orderVals[2] + orderVals[3]
fps = orderVals[4]
power = orderVals[5]/1000

data1 = [s1_total, s2_total]
data1stacked = {
    'Conv layers ': {'input': orderVals[0], 'inference': orderVals[1]},
    'Fully connected layers ': {'input': orderVals[2], 'inference': orderVals[3]}
}
data2 = [ fps, power]
labels1 = ["Conv layers", "Fully connected layers"]
labels2 = ["FPS", "Power"]

# fig, ax = plt.subplots()

# colors = ["tab:blue", "tab:orange", "tab:gray"]

x = np.arange(len(orders))  # the label locations
# width = 0.45  # the width of the bars
# multiplier = 0
# print(data1stacked)
# for colorindex, (stage, stack) in enumerate(data1stacked.items()):
#     offset = width * multiplier
#     bottom = np.zeros(6)
#     for height, data in stack.items():
#         color = colors[colorindex] if height != 'input' else colors[2]
#         rects = ax.bar(x + offset, data, width, bottom=bottom, color=color)
#         bottom += data
#         if height == 'inference':
#             ax.bar_label(rects, padding=3)
#     multiplier += 1

# ax.set_ylabel('Latency (ms)')
# ax.set_title('Latency by AlexNet part for various Pipe-ALL configurations (lower is better)')
# ax.set_xticks(x + 0.5*width, expanded_orders)
# lg1 = mpatches.Patch(color='tab:blue', label='Convolutional layers')
# lg2 = mpatches.Patch(color='tab:orange', label='Fully connected layers')
# lg3 = mpatches.Patch(color='tab:gray', label='Input time')
# plt.legend()
# ax.legend(handles=[lg1, lg2, lg3], loc='upper left', ncols=2)
# ax.set_ylim(0, 270)

# plt.show()

# ### new plot ##################################################################

fig, ax1 = plt.subplots()
width = 0.25
multiplier = 0

color = 'tab:red'
ax1.set_ylabel('FPS', color=color)  # we already handled the x-label with ax1
offset = width*0.9 * multiplier
rects = ax1.bar(x + offset, data2[0], width, label=labels2[0], color=color)
ax1.tick_params(axis='y', labelcolor=color)

multiplier += 1
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Power (W)', color=color)
offset = width*1.1 * multiplier
rects = ax2.bar(x + offset, data2[1], width, label=labels2[1], color=color)
ax1.bar_label(rects, padding=3)
ax2.tick_params(axis='y', labelcolor=color)
# multiplier += 1

ax1.set_title('FPS and Power by AlexNet part for various Pipe-ALL configurations')
ax1.set_xticks(x + 0.5*width, expanded_orders)
ax1.legend(loc='upper left', ncols=3)
ax1.set_ylim(0, 270)


# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.title("GPU on, modulating big frequency")
plt.show()

# ### new plot ##################################################################



# # c = Aggregator()
# # processOutputs = [f"test_results/single_components/big/{i}/adb_output.txt" for i in range(1,4)] + [f"test_results/single_components/gpu/{i}/adb_output.txt" for i in range(1,3)]
# # powerOutputs = [f"test_results/single_components/big/{i}/power_output.txt" for i in range(1,4)] + [f"test_results/single_components/gpu/{i}/power_output.txt" for i in range(1,3)]
# # c.bulkAggregate(processOutputs, powerOutputs)
# # print(c.averaged)


c = Aggregator()
c.aggregate("test_results/little_bottlenecking/adb_output.txt", None)
bnplotLabels, bnPlotVals = formatToPLT(c.split, [], ["pp1", "s1_inference", "s1_input", "s2_inference", "s2_input"])
# littleplotLabels = ['pp1', 's1_input', 's1_inference', 's2_input', 's2_inference']
pp = np.array([0,1,2,3,4,5,6,7])
little_inf = np.array([bnPlotVals[2][0], *bnPlotVals[4][1:]])
little_inp = np.array([bnPlotVals[1][0], *bnPlotVals[3][1:]])
stackeddata = {"input": little_inp, "inference": little_inf}

print(sum(little_inf[:5]))
print(sum(little_inf[5:]))
fig, ax = plt.subplots()

colors = ["tab:blue", "tab:orange", "tab:gray"]

# x = np.arange(len(little_inf))  # the label locations
width = 0.45  # the width of the bars
multiplier = 0
bottom = np.zeros(8)
for height, data in stackeddata.items():
    color = "tab:blue" if height != 'input' else "tab:gray"
    rects = ax.bar(pp, data, width, bottom=bottom, color=color)
    bottom += data
    if height == 'inference':
        ax.bar_label(rects, padding=3)

ax.set_ylabel('Latency (ms)')
ax.set_title('Latency by AlexNet part run by little')
# ax.set_xticks(pp)
lg1 = mpatches.Patch(color='tab:blue', label='Inference time')
# lg2 = mpatches.Patch(color='tab:orange', label='Fully connected layers')
lg3 = mpatches.Patch(color='tab:gray', label='Input time')
plt.legend()
ax.legend(handles=[lg1, lg3], loc='upper left', ncols=2)
ax.set_ylim(0, 270)

plt.show()

