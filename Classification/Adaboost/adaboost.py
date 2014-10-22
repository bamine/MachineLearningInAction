import numpy as np


def loadSimpleData():
    dataMat = np.matrix([[1, 2.1], [2, 1.1], [1.3, 1], [1, 1], [2, 1]])
    classLabels = [1, 1, -1, -1, 1]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArray, classLabels, D):
    dataMatrix = np.mat(dataArray)
    labelMatrix = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEstimator = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in xrange(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in xrange(-1, int(numSteps) + 1):
            for inequality in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedValues = stumpClassify(dataMatrix, i, threshVal, inequality)
                errorArray = np.mat(np.ones((m, 1)))
                errorArray[predictedValues == labelMatrix] = 0
                weightedError = D.T * errorArray
                print "split: dim %d - thresh: %.2f - thresh inequality: %s - the weighted error is %.3f" \
                      % (i, threshVal, inequality, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEstimator = predictedValues.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = threshVal
                    bestStump['ineq'] = inequality
    return bestStump, minError, bestClassEstimator


if __name__ == "__main__":
    dataMat, classLabels = loadSimpleData()
    D = np.mat(np.ones((5, 1)) / 5)
    buildStump(dataMat, classLabels, D)

