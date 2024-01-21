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

a = Aggregator("adbParser/adb_output.txt", "powerLogger/power_output.txt")
a.aggregate()

