from regression import *
from abalone import rssError


def regularize(xMat):  # regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)  #calc mean then subtract it off
    inVar = np.var(inMat, 0)  #calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat


def stageWise(x, y, eps=0.01, numIt=100):
    xMat = np.mat(x)
    yMat = np.mat(y).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    returnMat = np.zeros((numIt, n))
    for i in xrange(numIt):
        print ws.T
        lowestError = np.inf
        for j in xrange(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
            ws = wsMax.copy()
            returnMat[i, :] = ws.T
    return returnMat


if __name__ == "__main__":
    x, y = loadDataSet('abalone.txt')
    stageWise(x, y, 0.01, 200)

