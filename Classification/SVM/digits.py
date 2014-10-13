import numpy as np
from os import listdir
from smo import *


def img2vector(filename):
    returnVector = np.zeros((1, 1024))
    f = open(filename)
    for i in xrange(32):
        line = f.readline()
        for j in xrange(32):
            returnVector[0, 32 * i + j] = int(line[j])
    return returnVector


def loadImages(dirname):
    hwLabels = []
    trainingFileList = listdir(dirname)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in xrange(m):
        filenameStr = trainingFileList[i]
        filestr = filenameStr.split('.')[0]
        classNumStr = int(filestr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirname, filenameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    data, labels = loadImages('trainingDigits')
    b, alphas = smo(data, labels, 200, 0.0001, 10000, kTup)
    dataMat = np.mat(data)
    labelMat = np.mat(labels).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print "There are %d Support Vectors" % np.shape(sVs)[0]
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in xrange(m):
        kernelEval = kernelTransform(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labels[i]):
            errorCount += 1
    print "The training error rate is %f " % (float(errorCount) / m)
    data, labels = loadImages('testDigits')
    dataMat = np.mat(data)
    labelMat = np.mat(labels).transpose()
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in xrange(m):
        kernelEval = kernelTransform(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labels[i]):
            errorCount += 1
    print "The test error rate is %f " % (float(errorCount) / m)


if __name__ == "__main__":
    testDigits()
