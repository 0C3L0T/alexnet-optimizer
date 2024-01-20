
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
    def __init__(self, processOutputPath, powerOutputPath):
        self.tests = []
        self.processOutputPath = processOutputPath
        self.powerOutputPath = powerOutputPath

    def aggregate(self):
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
            avgPowerLabels = map(lambda x: 'avg' + x.capitalize(), powerLabels)
            peakPowerLabels = map(lambda x: 'peak' + x.capitalize(), powerLabels)
            avgPowerDict = dict(transpose([avgPowerLabels, powerAvgs]))
            peakPowerDict = dict(transpose([peakPowerLabels, powerPeaks]))

            output = perfDict | peakPowerDict | avgPowerDict
            self.tests.append(output)

        powerOutput.close()
        processOutput.close()


if __name__ == "__main__":
    a = Aggregator("adbParser/adb_output.txt", "powerLogger/power_output.txt")
    a.aggregate()
    print(a.tests)
