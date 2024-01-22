from measurementAggregator import transpose
import numpy as np

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

def reformat(tests):
    """
    [ [{inKey: inVal,}, {outKey: outVal,}], ] -> [ [[(inKey, inVal),], [(outKey,outVal),]], ]
    """
    return [list(map(lambda x: list(x.items()), test)) for test in tests]

def flatten(xss):
    """
    [[a, b], [c, d]] -> [a, b, c, d]
    [[(inKey, inVal),], [(outKey,outVal),]] -> [(inKey, inVal), (outKey,outVal),]

    """
    return [x for xs in xss for x in xs]

def merge(tests):
    """
    Removes the split input and output.
    [ [[(inKey, inVal),], [(outKey,outVal),]], ] -> [ [(inKey, inVal), (outKey,outVal),], ]
    """
    return [flatten(test) for test in tests]

def singleTestVariableIsolator(test, keeps):
    """
    Keeps only the desired keys in the test run data.
    """
    return list(filter(lambda y: y[0] in keeps, test))

def variableIsolator(tests, keeps):
    """
    maps singleTestVariableIsolator to each test run in tests.
    """
    return [singleTestVariableIsolator(test, keeps) for test in tests]

def outputBundle(tests):
    """
    extract the labels from the tests, and bundle the values in each seperate
    test to lists of data over time, ready to be used in matplotlib.
    """
    keys = transpose(tests[0])[0]
    vals = transpose([transpose(test)[1] for test in tests])
    return keys, vals

def formatToPLT(data, test_filter, keeps):
    """
    args:
        data: list of tests each with split input and output values.
            test_filter
    , Filter the tests to only those relevant, then
    filter the values to only those relevant, and output a format easy to be
    used in a script to plot the data.
    """
    tests_filtered = filter(InputValFilterCurry(test_filter), data)
    labels, vals = outputBundle(variableIsolator(merge(reformat(tests_filtered)), keeps))
    return labels, np.array(vals)
