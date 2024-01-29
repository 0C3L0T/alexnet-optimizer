import numpy as np
import matplotlib.pyplot as plt

BF_RANGE = 2208-500
LF_RANGE = 1800-500
bf1 = np.array([500, 1398, 2208])*1000
lf = np.array([500,1800])*1000
bf2 = np.array([500, 667, 1000, 1398, 1608, 1704, 1908, 2100, 2208], dtype=np.float64)*1000
y3 = np.array()

def scalebf(bf):
    return (bf-500) / BF_RANGE
def scalelf(lf):
    return (lf-500) / LF_RANGE
# y1 = np.array([0.12, 0.17, 0.19])
# y2 = np.array([0.09, 0.16, 0.19])
y1 = np.array([0.63,0.82,1.37])
y2 = np.array([0.56,0.59,0.64,0.73,0.78,0.81,0.91,1.09,1.24])
# y2 = np.array([0.])

# 0.05 +

# print(np.vstack((bf2,y1)).T)

SML_IDLE_DIFF = 0.01
BIG_IDLE_DIFF = 0.04

SML_UTIL_BASE = 0.05
BIG_UTIL_BASE = 0.10

SML_UTIL_SCALE = 0.09-SML_IDLE_DIFF
BIG_UTIL_SCALE = 0.11-BIG_IDLE_DIFF

# GPU_ON_BASE =
# {{500000, 0.56}, {667000, 0.59}, {1000000, 0.64}, {1398000, 0.73}, {1608000, 0.78}, {1704000, 0.81}, {1908000, 0.91}, {2100000, 1.09}, {2208000, 1.24}}
def bigamp(bf):
    # ans = (
    #       2.77904e-19 * bf**3
    #     - 8.72535e-13 * bf**2
    #     + 1.02561e-6  * bf
    #     + 0.22147
    # )
    bf = bf*1e-3
    ans = (
          2.77904e-10 * bf**3
        - 8.72535e-7  * bf**2
        + 0.00102561  * bf
        + 0.22147
    )
    # print("ans:", ans)
    return ans

def smlamp(lf):
    0.0000616413 * lf + 0.418258

# x = b2
# print(bf2[-1])
# print(bigamp(bf2[-1]))
# print(bigamp(2208000))
# print([bigamp(x) for x in bf2])
# print(bigamp(bf2))
# exit()

# def littleutilamp(lf,util):
#     return (SML_UTIL_BASE + SML_UTIL_SCALE*scalelf(lf)) * util
# def bigutilamp(bf,util):
#     return (BIG_UTIL_BASE + BIG_UTIL_SCALE*scalelf(bf)) * util

# def amp(bf, lf):
#  # base + util base + active scale + idle scale
#     ans = (
#         0.4
#         + littleutilamp(lf,1) + SML_IDLE_DIFF*scalelf(lf)
#         + bigutilamp(bf,1)    + BIG_IDLE_DIFF*scalebf(bf)
#     )
#     return ans

plt.plot(bf2, y2, label="small low")
plt.plot(bf1, y1, label="small high")
plt.plot(bf2, bigamp(bf2), label="lsq small low")
# plt.plot(bf2, amp(bf2, lf[0]), label="estim1")
# plt.plot(bf1, amp(bf1, lf[1]), label="estim2")
plt.legend()
plt.show()