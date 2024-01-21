import re


def transpose(l):
    return list(map(list, zip(*l)))

def mapIntOrFloatOrString(input):
    res = []

    for i in input:
        try:
            res.append(int(i))
        except ValueError:
            try:
                res.append(float(i))
            except ValueError:
                res.append(str(i))

    return res

def parseLine(line: str):
    line = re.sub(r'(?:(,) )|(?:(:) )|(?:(]) )|(?:([0-9])(?:V|mA|mW))',
                  r'\1\2\3\4',
                  line.strip('[ \n,'))
    time, *outputStr = re.split(r'\]|,', line)
    time = float(time)

    if len(outputStr) <= 1:
        return time, None

    outputT = transpose(map(lambda x: x.split(':'), outputStr))
    outputT[1] = mapIntOrFloatOrString(outputT[1])
    return time, outputT

class Aggregator():
    """
    Aggregator class is responsible for merging performance and power data into
    one data structure.

    Constructor args:
        processOutputPath (str): The file path of the process output data.
        powerOutputPath (str): The file path of the power output data.

    Attributes:
        tests (list): A list of dictionaries containing aggregated performance
        and power data.

    Default format of attribute "tests":
        list of dictionaries, each list is a test, and the dictionary contains
        data of several values measured or known from the test. These are:
        (known):
            Little Frequency  : int ∈ { 0, 500000, 667000, 1000000,
                                        1200000, 398000, 1512000,
                                        1608000, 1704000, 1800000 } (Hz)
            Big Frequency     : int ∈ { 0, 500000, 667000, 1000000,
                                        1200000, 1398000, 1512000,
                                        1608000, 1704000, 1800000,
                                        1908000, 2016000, 2100000,
                                        2208000 } (Hz)
            GPU On            : int ∈ {0, 1} (Bool)
            pp1               : int ∈ {0...8}         # "Partition Point"
            pp2               : int ∈ {0...8}
            order             : str ∈ {"a-b-c" | a,b,c ∈ {G, B, L}}
        (measured):
            duration          : float (s)
            s1 | s2 | s3:                    # "Stage n"
                _input        : float (s)
                _inference    : float (s)
            fps               : float (frames / s)
            latency           : float (s)    # MISSING IN DATA???
            avg | peak:
                Voltage       : float (V)
                Current       : float (mA)
                Power         : float (mW)
    """

    def __init__(self, processOutputPath, powerOutputPath):
        self.tests = []
        self.split = []
        self.processOutputPath = processOutputPath
        self.powerOutputPath = powerOutputPath

    def splitKnown(self):
        self.split = []
        for i, test in enumerate(self.tests):
            self.split.append([{},{}])
            for key,val in test.items():
                if isinstance(val, float):
                    self.split[i][1][key] = val
                else:
                    self.split[i][0][key] = val

    def aggregate(self, append=False, autoSplit=True):
        """
        Aggregates the performance and power data.

        Returns:
            None

        """
        if not append:
            self.tests = []

        processOutput = open(self.processOutputPath, "r")
        powerOutput = open(self.powerOutputPath, "r")

        endTime = 0
        lastPowerDataPoint = []
        powerLabels = []
        for line in processOutput:
            startTime = endTime
            endTime, perfData = parseLine(line)
            if perfData == None:
                continue
            perfDict = dict(transpose(perfData))
            perfDict['duration'] = endTime-startTime

            powerData = []
            if lastPowerDataPoint:
                powerData.append(powerDataPoint)
            while True:
                powerPoll = powerOutput.readline()
                if not powerPoll:
                    break
                time, (powerLabels, powerDataPoint) = parseLine(powerPoll)
                if time > endTime:
                    lastPowerDataPoint = powerDataPoint
                    break
                elif time < startTime:
                    continue
                powerData.append(powerDataPoint)

            powerDataT = transpose(powerData)
            powerAvgs = list(map(lambda x : sum(x)/len(x), powerDataT))
            powerPeaks = list(map(max, powerDataT))
            avgPowerLabels = map(lambda x: 'avg'+x.capitalize(), powerLabels)
            peakPowerLabels = map(lambda x: 'peak'+x.capitalize(), powerLabels)
            avgPowerDict = dict(transpose([avgPowerLabels, powerAvgs]))
            peakPowerDict = dict(transpose([peakPowerLabels, powerPeaks]))

            output = perfDict | peakPowerDict | avgPowerDict
            self.tests.append(output)

        powerOutput.close()
        processOutput.close()

        if autoSplit:
            self.splitKnown()


if __name__ == "__main__":
    a = Aggregator("adbParser/adb_output.txt", "powerLogger/power_output.txt")
    a.aggregate()
    print(a.split)
