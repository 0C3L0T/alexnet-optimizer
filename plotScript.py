from measurementGrapher import *
from measurementAggregator import Aggregator
import matplotlib.pyplot as plt
import numpy as np

a = Aggregator("adbParser/adb_output.txt", "powerLogger/power_output.txt")
a.aggregate()



littleplotLabels, littlePlotVals = formatToPLT(a.split, [("Big Frequency", 0), ("GPU On", 0)], ["fps", "avgPower", "Little Frequency"])
##### vals are: freq, pwr, fps #####

fig, ax1 = plt.subplots()
littleFreq = littlePlotVals[0]/1e6
pwr = littlePlotVals[2]/1000
fps = littlePlotVals[1]

color = 'tab:red'
ax1.set_xlabel('Little Frequency (GHz)')
ax1.set_ylabel('FPS', color=color)  # we already handled the x-label with ax1
ax1.plot(littleFreq, fps, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Power (W)', color=color)
ax2.plot(littleFreq, pwr, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title("Little cluster power and performance")
plt.show()

###
fpspw = fps/pwr
print(fps)
print(pwr)
print(fpspw)

plt.xlabel('Little Frequency (GHz)')
plt.ylabel('FPS/Watt')
plt.plot(littleFreq, fpspw)
plt.title("Little cluster efficiency ")

plt.show()
