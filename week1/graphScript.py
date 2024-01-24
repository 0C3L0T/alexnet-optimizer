from measurementTools import *
from measurementAggregator import Aggregator
import matplotlib.pyplot as plt
import numpy as np

# a = Aggregator()
# a.aggregate("test_results/single_components/adb_output.txt",
#             "test_results/single_components/power_output.txt",
#             append=True)
# a.aggregate("")

# processOutputs = [f"test_results/single_components/big/{i}/adb_output.txt" for i in range(1,4)] + [f"test_results/single_components/gpu/{i}/adb_output.txt" for i in range(1,3)]
# powerOutputs = [f"test_results/single_components/big/{i}/power_output.txt" for i in range(1,4)] + [f"test_results/single_components/gpu/{i}/power_output.txt" for i in range(1,3)]
# a.bulkAggregate(processOutputs, powerOutputs)
a = Aggregator()
a.aggregate("test_results/single_components/adb_output.txt",
            "test_results/single_components/power_output.txt")

single = a.split
# print(single)
del a


b = Aggregator()
b.aggregate("test_results/order/adb_output.txt",
            "test_results/order/power_output.txt")

multi = b.split
del b

### new plot ##################################################################

littleplotLabels, littlePlotVals = formatToPLT(single, [("Big Frequency", 0), ("GPU On", 0)], ["fps", "avgPower", "Little Frequency"])

##### vals are: freq, pwr, fps #####

fig, ax1 = plt.subplots()
littleFreq = littlePlotVals[0]/1e6
littlePwr = littlePlotVals[2]/1000
littleFps = littlePlotVals[1]

color = 'tab:red'
ax1.set_xlabel('Core clock (GHz)')
ax1.set_ylabel('FPS', color=color)  # we already handled the x-label with ax1
ax1.plot(littleFreq, littleFps, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Power (W)', color=color)
ax2.plot(littleFreq, littlePwr, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title("Little cluster power and performance")
plt.show()

### new plot ##################################################################


littlefpspw = littleFps/littlePwr
# print(littleFps)
# print(littlePwr)
# print(littlefpspw)

plt.xlabel('Core clock (GHz)')
plt.ylabel('FPS/Watt')
plt.plot(littleFreq, littlefpspw)
plt.title("Little cluster efficiency")

plt.show()

### new plot ##################################################################

bigplotLabels, bigPlotVals = formatToPLT(single, [("Little Frequency", 0), ("GPU On", 0)], ["fps", "avgPower", "Big Frequency"])

bigFreq = bigPlotVals[0]/1e6
bigPwr = bigPlotVals[2]/1000
bigFps = bigPlotVals[1]

bigfpspw = bigFps/bigPwr

plt.xlabel('Core clock (GHz)')
plt.ylabel('FPS/Watt')
plt.plot(littleFreq, littlefpspw, label="Little cluster")
plt.plot(bigFreq, bigfpspw, label="Big cluster")
plt.title("Big and little cluster efficiency ")
plt.legend()

plt.show()

### new plot ##################################################################

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Core clock (GHz)')
ax1.set_ylabel('FPS', color=color)  # we already handled the x-label with ax1
ax1.plot(bigFreq, bigFps, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Power (W)', color=color)
ax2.plot(bigFreq, bigPwr, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title("Big cluster power and performance")
plt.show()

### new plot ##################################################################

gpuPlotLabels, gpuPlotVals = formatToPLT(single, [("GPU On", 1)], ["fps", "avgPower", "Big Frequency"])

bigfreqgpu = gpuPlotVals[0]/1e6
fpsgpu = gpuPlotVals[1]
pwrgpu = gpuPlotVals[2]/1000

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Core clock (GHz)')
ax1.set_ylabel('FPS', color=color)  # we already handled the x-label with ax1
ax1.plot(bigfreqgpu, fpsgpu, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Power (W)', color=color)
ax2.plot(bigfreqgpu, pwrgpu, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title("GPU on, modulating big frequency")
plt.show()

# c = Aggregator()
# processOutputs = [f"test_results/single_components/big/{i}/adb_output.txt" for i in range(1,4)] + [f"test_results/single_components/gpu/{i}/adb_output.txt" for i in range(1,3)]
# powerOutputs = [f"test_results/single_components/big/{i}/power_output.txt" for i in range(1,4)] + [f"test_results/single_components/gpu/{i}/power_output.txt" for i in range(1,3)]
# c.bulkAggregate(processOutputs, powerOutputs)
# print(c.averaged)
