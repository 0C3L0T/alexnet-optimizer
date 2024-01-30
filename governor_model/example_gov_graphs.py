import numpy as np
import matplotlib.pyplot as plt

def transpose(l):
    return list(map(list, zip(*l)))


with open("example_governor_measurements.txt", "r") as f:
    measures = f.read().strip().split("\n\n")

data = [m.split("\n") for m in measures]
data_ = []
for r in data[:]:
    r[1] = float(r[1].split(", ")[1])
    r[0] = [int(b.split("=")[1]) for b in r[0].split(", ")]
    data_.append([*r[0], r[1]])
data = data_
# print(data)
g1 = np.array(data[:4]).T
g2 = np.flip(np.array(data[5:]).T, axis=1)
# print(g2)
# print(np.flip(g2))


lat, fps, duration = g2
ga_duration = np.array([34.228, 34.507, 34.520, 34.533, 34.093, 33.442])
plt.plot(lat, duration, label="Example Governor")
plt.plot(lat, ga_duration, label="Genetic Algorithm")
plt.xlabel("Target Latency")
plt.ylabel("Time to achieve target")
plt.xlim([lat[0],lat[-1]])
plt.legend()
plt.show()

# labels1 = ["Conv layers", "Fully connected layers"]


fig, ax1 = plt.subplots()
width = 20
multiplier = 0

color = 'tab:purple'
ax1.set_ylabel('Duration (s)')  # we already handled the x-label with ax1
ax1.set_xlabel('Latency target (ms)')
offset = width*0.9 * multiplier
rects = ax1.bar(lat[:-1] + offset, duration[:-1], width, label="Example Governor", color=color)
ax1.bar(lat[-1:] + offset, duration[-1:], width, label="Example Governor (no solution)", color="tab:gray")
# ax1.tick_params(axis='y',)

multiplier += 1
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
# ax1.set_ylabel('', color=color)
noffset = width*1.1 * multiplier
rects = ax1.bar(lat + noffset, ga_duration, width, label="Genetic Algorithm", color=color)
# ax1.bar_label(rects, padding=3)
# ax1.tick_params(axis='y', labelcolor=color)
# multiplier += 1

ax1.set_title('Solving time for various latency targets. FPS target = 3')
# ax1.set_xticks(x + 0.5*width, )
ax1.legend(loc='upper right', ncols=1)
# ax1.set_ylim(0, 270)

plt.show()



lat, fps, duration = g1
ga_duration = np.array([11.858, 12.070, 11.975, 11.899])
# print()
# print(lat, fps, duration)

# plt.plot(fps, duration, label="Example Governor")
# plt.plot(fps, ga_duration, label="Genetic Algorithm")
# plt.xlabel("Target FPS")
# plt.ylabel("Time to achieve target")
# plt.legend()
# plt.show()
fig, ax1 = plt.subplots()
width = 1
multiplier = 0

color = 'tab:purple'
ax1.set_ylabel('Duration (s)')  # we already handled the x-label with ax1
ax1.set_xlabel('FPS target')
offset = width*0.9 * multiplier
rects = ax1.bar(fps + offset, duration, width, label="Example Governor", color=color)
# ax1.tick_params(axis='y',)

multiplier += 1
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
# ax1.set_ylabel('', color=color)
noffset = width*1.1 * multiplier
rects = ax1.bar(fps + noffset, ga_duration, width, label="Genetic Algorithm", color=color)
# ax1.bar(fps[-1:] + offset, duration[-1:], width, label="Example Governor (no solution)", color="tab:gray")

# ax1.bar_label(rects, padding=3)
# ax1.tick_params(axis='y', labelcolor=color)
# multiplier += 1

ax1.set_title('Solving time for various FPS targets. Latency target = 200')
# ax1.set_xticks(x + 0.5*width, )
ax1.legend(loc='upper left', ncols=2)
ax1.set_ylim(0, 400)

plt.show()