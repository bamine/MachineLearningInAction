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
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
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


def isTree(obj):
    return type(obj).__name__ == 'treeNode'


def getMean(tree):
    if isTree(tree.right):
        tree.right = getMean(tree.right)
    if isTree(tree.left):
        tree.left = getMean(tree.left)
    return (tree.left + tree.right) / 2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree.right) or isTree(tree.left):
        lSet, rSet = binSplitData(testData, tree.feature, tree.value)
        if isTree(tree.left):
            tree.left = prune(tree.left, lSet)
        if isTree(tree.right):
            tree.right = prune(tree.right, rSet)
    if not isTree(tree.left) and not isTree(tree.right):
        lSet, rSet = binSplitData(testData, tree.feature, tree.value)
        errorNoMerge = np.sum(np.square(lSet[:, -1] - tree.left)) + np.sum(np.square(rSet[:, -1] - tree.right))
        treeMean = (tree.left + tree.right) / 2.0
        errorMerge = np.sum(np.square(testData[:, -1] - treeMean))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__ == "__main__":
    data = loadDataMat('ex2.txt')
    mat2 = np.mat(data)
    myTree = createTree(mat2, ops=(0, 1))
    test = loadDataMat('ex2test.txt')
    mat2test = np.mat(test)
    print prune(myTree, mat2test)



