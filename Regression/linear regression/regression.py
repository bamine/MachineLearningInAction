import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in xrange(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standardRegression(x, y):
    xMat = np.mat(x)
    yMat = np.mat(y).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print "this matrix is singular, cannot invert"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def locallyWeightedLinearRegression(testPoint, x, y, k=1.0):
    xMat = np.mat(x)
    yMat = np.mat(y).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in xrange(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot invert"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArray, x, y, k=1.0):
    m = np.shape(testArray)[0]
    yHat = np.zeros(m)
    for i in xrange(m):
        yHat[i] = locallyWeightedLinearRegression(testArray[i], x, y, k)
    return yHat


if __name__ == "__main__":
    x, y = loadDataSet('ex0.txt')
    print locallyWeightedLinearRegression(x[0], x, y, 1.0)
    print locallyWeightedLinearRegression(x[0], x, y, 0.01)
    yHat = lwlrTest(x, x, y, 0.01)
    xMat = np.mat(x)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1].flatten().A[0], np.mat(yHat[srtInd]).T.flatten().A[0])
    ax.scatter(xMat[:, 1].flatten().A[0], np.mat(y).T.flatten().A[0], s=2, c='red')
    plt.show()
