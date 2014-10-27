from regression import *
import numpy as np


def rssError(y, yHat):
    return ((y - yHat) ** 2).sum()


if __name__ == "__main__":
    X, Y = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(X[:99], X[:99], Y[:99], 0.1)
    yHat1 = lwlrTest(X[:99], X[:99], Y[:99], 1)
    yHat10 = lwlrTest(X[:99], X[:99], Y[:99], 10)
    print "error for k=0.1 is ", rssError(Y[:99], yHat01.T)
    print "error for k=1 is ", rssError(Y[:99], yHat1.T)
    print "error for k=10 is ", rssError(Y[:99], yHat10.T)