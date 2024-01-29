#################### PLACEHOLDER DUMMY POWER PREDICTOR ####################

LAYERS = 8

freqLevels = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000,
              1704000, 1800000, 1908000, 2016000, 2100000, 2208000]

def predict_power(chromosome):
    pp1 = chromosome[0].layers
    pp2 = pp1 + chromosome[1].layers
    B_size = pp1
    L_size = pp2-pp1
    G_size = LAYERS-pp2
    bfreq = freqLevels[chromosome[0].frequency_level]/1000
    lfreq = freqLevels[chromosome[2].frequency_level]/1000
    return (L_size*lfreq + B_size*bfreq*3 + G_size*bfreq*2)