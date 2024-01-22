import re

def reformat(tests):
    """
    [ [{inKey: inVal,}, {outKey: outVal,}], ] -> [ [[(inKey, inVal),], [(outKey,outVal),]], ]
    """
    return [list(map(lambda x: list(x.items()), test)) for test in tests]

def InputValFilterCurry(kv):
    """
    Filter auxiliary function for filtering tests only with desired input
    values.

    args:
        kv: List of iterables of one key and one or more values where each key
            is a data label and the values are the permitted values for the
            list to pass the filter check.

    usage:
        filter(InputValFilterCurry([
                    ("order", "G-L-B", "G-B-L"),
                    ("GPU On", 1),
                    ("pp2", *range(4,9))
                ]), a.split)
    """
    def filterFunc(test):
        inputVals = test[0] # first index is input
        for k,*v in kv:
            if inputVals[k] not in v:
                return False
        return True
    return filterFunc

def OutputValFilterCurry(kv):
    """
    Filter auxiliary function for filtering tests only with desired output
    values.

    args:
        kv: Key function pair where each key is a data label and each function
            is a one argument function judging wether the value associated with
            the key prevents the list from passing the filter check.
            (function returns False to prevent the list from passing).

    usage:
        filter(outputValFilterCurry([
                    ("avgPower", lambda x: x <= 2000.0),
                    ("fps", lambda x: x > 5.0)
                ]), a.split)
    """
    def filterFunc(test):
        outputVals = test[1] # second index is output
        for k,v in kv:
            if not v(outputVals[k]):
                return False
        return True
    return filterFunc

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

    def __init__(self):
        self.tests = []
        self.split = []
        self.averaged = []

    def averageMeasurements(self):
        if self.split == []:
            return

        passedsigs = []
        for test in self.split:
            if test[0].__repr__() in passedsigs:
                continue
            passedsigs.append(test[0].__repr__())
            similar = list(filter(InputValFilterCurry(test[0].items()), self.split))
            keys = similar[0][1].keys()
            results = transpose([transpose(simTest[1])[1] for simTest in reformat(similar)])
            resultsAvg = [sum(vals)/len(vals) for vals in results]
            reconstructed = [test[0], dict(transpose([keys, resultsAvg]))]
            self.averaged.append(reconstructed)

    def splitKnown(self):
        self.split = []
        for i, test in enumerate(self.tests):
            self.split.append([{},{}])
            for key,val in test.items():
                if isinstance(val, float):
                    self.split[i][1][key] = val
                else:
                    self.split[i][0][key] = val

    def bulkAggregate(self, processOutputPaths, powerOutputPaths):
        if len(processOutputPaths) != len(powerOutputPaths):
            raise ValueError
        for i in range(len(processOutputPaths)):
            self.aggregate(processOutputPaths[i], powerOutputPaths[i], append=True)
        self.averageMeasurements()

    def aggregate(self, processOutputPath, powerOutputPath, append=False, autoSplit=True):
        """
        Aggregates the performance and power data.

        Returns:
            None

        """
        if not append:
            self.tests = []

        processOutput = open(processOutputPath, "r")
        powerOutput = open(powerOutputPath, "r")

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


# if __name__ == "__main__":
#     a = Aggregator()
#     a.aggregate("adbParser/adb_output.txt", "powerLogger/power_output.txt")
#     print(a.split)
