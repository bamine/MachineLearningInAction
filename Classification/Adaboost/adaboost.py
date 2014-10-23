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
                # print "split: dim %d - thresh: %.2f - thresh inequality: %s - the weighted error is %.3f" \
                #      % (i, threshVal, inequality, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEstimator = predictedValues.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = threshVal
                    bestStump['ineq'] = inequality
    return bestStump, minError, bestClassEstimator


def adaBoostTrain(dataSet, classLabels, numIt=40):
    weakClassArray = []
    m = np.shape(dataSet)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in xrange(numIt):
        bestStump, error, classEst = buildStump(dataSet, classLabels, D)
        # print "D :",D.T
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArray.append(bestStump)
        #print "classEst : ",classEst.T
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        #print "aggClassEst :",aggClassEst.T
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "Total errors :", errorRate, "\n"
        if errorRate == 0.0:
            break
    return weakClassArray


def adaClassify(data, classifierArray):
    dataMatrix = np.mat(data)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for classifier in classifierArray:
        classEst = stumpClassify(dataMatrix, classifier['dim'], classifier['threshold'], classifier['ineq'])
        aggClassEst += classifier['alpha'] * classEst
        # print aggClassEst
    return np.sign(aggClassEst)


if __name__ == "__main__":
    dataMat, classLabels = loadSimpleData()
    D = np.mat(np.ones((5, 1)) / 5)
    classifierArray = adaBoostTrain(dataMat, classLabels)
    print classifierArray
    print adaClassify([0, 0], classifierArray)

