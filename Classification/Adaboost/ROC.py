import matplotlib.pyplot as plt
import numpy as np
from adaboost import *
from horse import *


def plotROC(predStrengths, labels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = np.sum(np.array(labels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(labels) - numPosClas)
    sortedIndices = predStrengths.argsort()
    fig = plt.figure()
    ax = plt.subplot(111)
    for index in sortedIndices.tolist()[0]:
        if labels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost Horse Colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print "The Area under the ROC curve is ", ySum * xStep


if __name__ == "__main__":
    dataArray, labelArray = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrain(dataArray, labelArray, 40)
    plotROC(aggClassEst.T, labelArray)
