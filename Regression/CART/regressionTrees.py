import numpy as np
from tree import treeNode


def loadDataMat(filename):
    dataMat = []
    f = open(filename)
    for line in f.readlines():
        currentLine = line.strip().split('\t')
        floatLine = map(float, currentLine)
        dataMat.append(floatLine)
    return dataMat


def binSplitData(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1


def regLeaf(dataSet):
    return np.mean(dataSet[:, 1])


def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist())[0]) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in xrange(n - 1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitData(dataSet, featIndex, splitVal)
            if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if S - bestS < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitData(dataSet, bestIndex, bestValue)
    if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    regTree = treeNode()
    regTree.feature = feat
    regTree.value = val
    lSet, rSet = binSplitData(dataSet, feat, val)
    regTree.left = createTree(lSet, leafType, errType, ops)
    regTree.right = createTree(rSet, leafType, errType, ops)
    return regTree



