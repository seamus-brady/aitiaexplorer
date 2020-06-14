"""
TBD
"""

class SimulatedData1Graph:
    """
    This is the graph for the data file simulated_data_1.txt.
    This data was created using Tetrad 6.7.1-0 with the following parameters:

    Bayes net simulation using Graph constructed by adding random forward edges
    graphsDropdownPreference = Random Foward DAG
    simulationsDropdownPreference = Bayes Net
    numMeasures = 10
    numLatents = 0
    avgDegree = 2
    maxDegree = 100
    maxIndegree = 100
    maxOutdegree = 100
    connected = false
    minCategories = 3
    maxCategories = 3
    numRuns = 1
    differentGraphs = false
    randomizeColumns = false
    sampleSize = 1000
    saveLatentVars = false
    coefLow = 0.2
    coefHigh = 0.7
    covLow = 0.0
    covHigh = 0.0
    varLow = 1.0
    varHigh = 3.0
    coefSymmetric = true
    covSymmetric = true
    measurementVariance = 0.0
    standardize = false
    scaleFreeAlpha = 0.05
    scaleFreeBeta = 0.9
    scaleFreeDeltaIn = 3
    scaleFreeDeltaOut = 3
    probCycle = 1.0
    probTwoCycle = 0.0
    numStructuralNodes = 3
    numStructuralEdges = 3
    measurementModelDegree = 5
    latentMeasuredImpureParents = 0
    measuredMeasuredImpureParents = 0
    measuredMeasuredImpureAssociations = 0
    """

    def __init__(self):
        pass

    @staticmethod
    def nodes(self):
        return ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"]

    @staticmethod
    def edges():
        return [("X1","X10"), ("X2","X4"), ("X2","X6"), ("X4","X6"), ("X4","X7"), ("X4","X8"), ("X4","X9"), ("X5","X8"), ("X6","X9"), ("X8","X9")]