import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    f = open(filename)
    numberOfLines = len(f.readlines())
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    labelEncoding = {"largeDoses": 1, "smallDoses": 2, "didntLike": 3}
    f = open(filename)
    for index, line in enumerate(f.readlines()):
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(labelEncoding[listFromLine[-1]])
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = (dataSet - np.tile(minVals, (m, 1))) / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def visualizeDatingData(datingDataMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.show()

def classifiy0(X, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(X, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = np.sqrt(sqDistances)
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in xrange(k):
        voteILabel = labels[sortedDistIndices[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def datingClassTest():
    ratio = 0.1
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * ratio)
    errorCount = 0.0
    for i in xrange(numTestVecs):
        classifierResult = classifiy0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 2)
        print "The classifier answer : %d - The real answer : %d" % (classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "The total error rate for the classifier is : %f" % (errorCount / numTestVecs)


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("Percentage of time spent playing video games ?"))
    ffMiles = float(raw_input("Frequent flier miles earned per year ?"))
    iceCream = float(raw_input("Liters of ice cream consumes per year ?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    X = (np.array([ffMiles, percentTats, iceCream]) - minVals) / ranges
    classifierResult = classifiy0(X, normMat, datingLabels, 3)
    print "You will probably like that person: ", resultList[classifierResult - 1]


if __name__ == "__main__":
    classifyPerson()
