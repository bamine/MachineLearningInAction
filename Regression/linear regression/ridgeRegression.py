from regression import *
from abalone import rssError


def ridgeRegression(x, y, lam=0.2):
    xTx = x.T * x
    denom = xTx + np.eye(np.shape(x)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot compute inverse"
        return
    ws = denom.I * (x.T * y)
    return ws


def ridgeTest(x, y):
    xMat = np.mat(x)
    yMat = np.mat(y).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in xrange(numTestPts):
        ws = ridgeRegression(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


if __name__ == "__main__":
    X, Y = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

