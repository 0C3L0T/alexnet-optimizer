from measurementAggregator import transpose
import numpy as np

def InputValFilterCurry(kv):
    """
    usage:
        filter(InputValFilterCurry([
                    ("order", "G-L-B", "G-B-L"),
                    ("GPU On", 1),
                    ("pp2", *range(4,9))
                ]), a.split)
    """
    def filterFunc(test):
        inputVals = test[0]
        for k,*v in kv:
            if inputVals[k] not in v:
                return False
        return True
    return filterFunc

def OutputValFilterCurry(kv):
    """
    usage:
        filter(outputValFilterCurry([
                    ("avgPower", lambda x: x <= 2000.0),
                    ("fps", lambda x: x > 5.0)
                ]), a.split)
    """
    def filterFunc(test):
        outputVals = test[1]
        for k,v in kv:
            if not v(outputVals[k]):
                return False
        return True
    return filterFunc

def reFormat(tests):
    """
    [[inDict, outDict],] -> [ [[(inKey, inVal),], [(outKey,outVal),]], ]
    """
    return [list(map(lambda x: list(x.items()), test)) for test in tests]

def flatten(xss):
    return [x for xs in xss for x in xs]

def merge(tests):
    return [flatten(test) for test in tests]

def singleTestVariableIsolator(test, keeps):
    # print(test)
    return list(filter(lambda y: y[0] in keeps, test))

def variableIsolator(tests, keeps):
    # print(tests)
    return [singleTestVariableIsolator(test, keeps) for test in tests]

def outputBundle(tests):
    keys = transpose(tests[0])[0]
    vals = transpose([transpose(test)[1] for test in tests])
    return keys, vals

def formatToPLT(data, test_filter, keeps):
    tests_filtered = filter(InputValFilterCurry(test_filter), data)
    labels, vals = outputBundle(variableIsolator(merge(reFormat(tests_filtered)), keeps))
    return labels, np.array(vals)
