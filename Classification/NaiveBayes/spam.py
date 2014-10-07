import numpy as np
import re
from bayes import *
import random


def parseText(string):
    tokens = re.split(r'\W*', string)
    return [tok.lower() for tok in tokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in xrange(1, 26):
        wordList = parseText(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = parseText(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = trainNaiveBayes0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifiyNB(np.array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
            print 'missclassified : %s' % docList[docIndex]
    print 'the error rate is %f' % (float(errorCount) / len(testSet))


if __name__ == "__main__":
    spamTest()




