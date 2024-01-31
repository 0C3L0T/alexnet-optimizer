LEVELS = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000,
              1704000, 1800000, 1908000, 2016000, 2100000, 2208000]

BASELINE_AMPS = 0.45

BF_BASE  = 0.080  # bigamp(500000)
LF_BASE  = 0.055  # smlamp(500000)
BF_RANGE = 0.673  # bigamp(2208000)-bigamp(500000)
LF_RANGE = 0.091  # smlamp(1800000)-smlamp(500000)

BIG_GPU_SUPP_FACTOR = 0.13

GPU_AMP = 0.12  # GPU power is constant since we can't control the frequency.

# {{500000, 0.56}, {667000, 0.59}, {1000000, 0.64}, {1398000, 0.73}, {1608000, 0.78}, {1704000, 0.81}, {1908000, 0.91}, {2100000, 1.09}, {2208000, 1.24}}
def bigamp(bf):
    bf = bf*1e-3
    ans = (
          2.77904e-10 * bf**3
        - 8.72535e-7  * bf**2
        + 0.00102561  * bf
        - 0.25
    )
    return ans

def smlamp(lf):
    lf = lf*1e-3
    return 7.0e-5 * lf + 0.02

def scalebf(bf):
    return (bigamp(bf)-BF_BASE) / BF_RANGE

def scalelf(lf):
    return (smlamp(lf)-LF_BASE) / LF_RANGE

def totalwatts(lf,bf,butil,gutil,lutil):
    return (
        BASELINE_AMPS
        + smlamp(lf)*lutil
        + bigamp(bf)*butil
        + BIG_GPU_SUPP_FACTOR*scalebf(bf) * max(0,gutil-butil)
        + GPU_AMP*gutil
    ) * 5.0

def predict_power(chromosome, l1util, l2util, l3util):
    bfreq = LEVELS[chromosome[0].frequency_level]
    lfreq = LEVELS[chromosome[2].frequency_level]
    return totalwatts(lfreq, bfreq, l1util, l2util, l3util)
