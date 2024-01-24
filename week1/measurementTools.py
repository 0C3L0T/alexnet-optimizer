from measurementAggregator import transpose, InputValFilterCurry, reformat
import numpy as np

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
    tests_filtered = list(filter(InputValFilterCurry(test_filter), data))
    if not len(tests_filtered):
        print("No matching tests")
    labels, vals = outputBundle(variableIsolator(merge(reformat(tests_filtered)), keeps))
    return labels, np.array(vals)
