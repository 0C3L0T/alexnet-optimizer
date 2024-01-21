from measurementAggregator import Aggregator, transpose

def InputValFilterCurry(kv):
    """
    usage:
        filter(InputValFilterCurry([
                    ("order", G-L-B),
                    ("GPU On", 1)
                ]), a.splitted)
    """
    def filterFunc(test):
        inputVals = test[0]
        for k,v in kv:
            if inputVals[k] != v:
                return False
        return True
    return filterFunc

def formatForPLT(results):
    """
    [inDict, outDict] -> [[inKeyList, inValList],[outKeyList,outValList]]
    """
    return list(map(lambda x: transpose(x.items()), results))

a = Aggregator("adbParser/adb_output.txt", "powerLogger/power_output.txt")
a.aggregate()

