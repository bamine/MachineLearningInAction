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


if __name__ == "__main__":
    x, y = loadDataSet('ex0.txt')
    ws = standardRegression(x, y)
    xMat = np.mat(x)
    yMat = np.mat(y)
    yHat = xMat * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1].flatten().A[0], yHat[:, 0].flatten().A[0])
    print np.corrcoef(yHat.T, yMat)
    plt.show()

