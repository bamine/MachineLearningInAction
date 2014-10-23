from adaboost import *


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


if __name__ == "__main__":
    dataArray, labelArray = loadDataSet('horseColicTraining2.txt')
    classifierArray = adaBoostTrain(dataArray, labelArray, 20)
    testArray, testLabelArray = loadDataSet('horseColicTest2.txt')
    predictions = adaClassify(testArray, classifierArray)
    errors = np.mat(np.ones((67, 1)))
    print errors[predictions != np.mat(testLabelArray).T].sum() / 67.0
