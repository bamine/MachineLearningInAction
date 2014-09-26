import numpy as np
from os import listdir
from kNN import *


def img2vector(filename):
    returnVector = np.zeros((1, 1024))
    f = open(filename)
    for i in xrange(32):
        line = f.readline()
        for j in xrange(32):
            returnVector[0, 32 * i + j] = int(line[j])
    return returnVector


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in xrange(m):
        fileName = trainingFileList[i]
        file = fileName.split('.')[0]
        classNumStr = int(file.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileName)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in xrange(mTest):
        fileName = testFileList[i]
        file = fileName.split('.')[0]
        classNumStr = int(file.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileName)
        classifierResult = classifiy0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "The classifier answer : %d - The real answer : %d" % (classifierResult, classNumStr)
        if classifierResult != classNumStr:
            errorCount += 1.0
    print "\n the total number of errors is %d" % errorCount
    print "\n the total error rate is %f" % (errorCount / float(mTest))


if __name__ == "__main__":
    handwritingClassTest()