import numpy as np
import matplotlib.pyplot as plt
import random

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        a = line.strip().split()
        dataMat.append([1.0, float(a[0]), float(a[1])])
        labelMat.append(int(a[2]))
    return dataMat, labelMat


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def gradientAscent(dataMat, labels):
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(labels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxEpochs = 500
    weights = np.ones((n, 1))
    for k in xrange(maxEpochs):
        h = sigmoid(dataMatrix * weights)
        error = labelMatrix - h
        weights += alpha * dataMatrix.transpose() * error
    return weights


def StochasticGradientAscent(dataMat, labels, iter=150):
    dataMatrix = np.array(dataMat)
    labelMatrix = np.array(labels)
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for k in xrange(iter):
        for i in xrange(m):
            h = sigmoid(sum(dataMatrix[i] * weights))
            error = labelMatrix[i] - h
            weights += alpha * dataMatrix[i] * error
    return weights


def StochasticGradientAscent2(dataMat, labels, iter=150):
    dataMatrix = np.array(dataMat)
    labelMatrix = np.array(labels)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for k in xrange(iter):
        dataIndex = range(m)
        for i in xrange(m):
            alpha = 4 / (1.0 + k + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[i] * weights))
            error = labelMatrix[i] - h
            weights += alpha * dataMatrix[i] * error
            del (dataIndex[randIndex])
    return weights


def plotBestFit(weights):
    data, labels = loadDataSet()
    dataArray = np.array(data)
    n = np.shape(dataArray)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in xrange(n):
        if int(labels[i]) == 1:
            xcord1.append(dataArray[i, 1])
            ycord1.append(dataArray[i, 2])
        else:
            xcord2.append(dataArray[i, 1])
            ycord2.append(dataArray[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classify(x, weights):
    prob = sigmoid(sum(x * weights))
    if prob < 0.5:
        return 0.0
    else:
        return 1.0


def colicTest():
    train = open('horseColicTraining.txt')
    test = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in train.readlines():
        l = line.strip().split('\t')
        array = []
        for i in xrange(21):
            array.append(float(l[i]))
        trainingSet.append(array)
        trainingLabels.append(float(l[21]))
    trainWeights = StochasticGradientAscent2(np.array(trainingSet), trainingLabels, 5000)
    errorCount = 0
    numTestVec = 0.0
    for line in test.readlines():
        numTestVec += 1.0
        l = line.strip().split('\t')
        array = []
        for i in xrange(21):
            array.append(float(l[i]))
        if int(classify(np.array(array), trainWeights)) != int(l[21]):
            errorCount += 1
        errorRate = float(errorCount) / numTestVec
    print "The error rate of the classifier is %f" % errorRate
    return errorRate


if __name__ == "__main__":
    x = colicTest()



