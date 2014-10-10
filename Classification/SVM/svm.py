import numpy as np
import random


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    f = open(filename)
    for line in f.readlines():
        array = line.strip().split('\t')
        dataMat.append([float(array[0]), float(array[1])])
        labelMat.append(float(array[2]))
    return dataMat, labelMat


def selectRand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataMat, labelMat, C, toler, maxIter):
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(labelMat).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in xrange(m):
            fXi = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMatrix[i])
            if (labelMatrix[i] * Ei < -toler and alphas[i] < C) or (labelMatrix[i] * Ei > toler and alphas[i] > 0):
                j = selectRand(i, m)
                fXj = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMatrix[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = max(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = max(C, alphas[j] + alphas[i])
                if L == H:
                    print "L==H"
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[
                                                                                                            j,
                                                                                                            :] * dataMatrix[
                                                                                                                 j, :].T
                if eta >= 0:
                    print "eta >= 0"
                    continue
                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print "j is not moving enough"
                    continue
                alphas[i] += labelMatrix[j] * labelMatrix[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if alphas[i] > 0 and alphas[i] < C:
                    b = b1
                elif alphas[j] > 0 and alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iteration: %d - i: %d - pairs changed: %d" % (iter, i, alphaPairsChanged)
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print "*** iteration number : %d" % iter
    return b, alphas


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet("testSet.txt")
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)





















