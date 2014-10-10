import numpy as np
import random
from smoSimple import *


class optStruct(object):
    def __init__(self, dataMat, labels, C, toler, kTup=['lin']):
        self.X = dataMat
        self.labels = labels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMat)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in xrange(self.m):
            self.K[:, i] = kernelTransform(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labels).T * oS.K[:, k]) + oS.b
    Ek = fXk - float(oS.labels[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectRand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labels[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or (oS.labels[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labels[i] != oS.labels[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] - oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print "L==H"
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print "eta >= 0"
            return 0
        oS.alphas[j] -= oS.labels[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print "j not moving enough"
            return 0
        oS.alphas[i] += oS.labels[j] * oS.labels[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labels[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labels[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labels[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labels[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
            oS.b = b1
        elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smo(dataMat, labels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMat), np.mat(labels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in xrange(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print "full set - iter: %d - i: %d - pairs changed: %d" % (iter, i, alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print "non bound - iter: %d - i: %d - pairs changed: %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print "iteration number: %d" % iter
    return oS.b, oS.alphas


def calculateWs(alphas, data, labels):
    X = np.mat(data)
    labelMat = np.mat(labels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in xrange(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def kernelTransform(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in xrange(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston we have a problem ... The kernel is not recognized')
    return K


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet("testSet.txt")
    b, alphas = smo(dataMat, labelMat, 0.6, 0.001, 40)
    ws = calculateWs(alphas, dataMat, labelMat)
    data = np.mat(dataMat)
    print data[0] * np.mat(ws) + b


































