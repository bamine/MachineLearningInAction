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


if __name__ == "__main__":
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    visualizeDatingData(datingDataMat)
