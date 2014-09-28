import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction', va='center',
                            ha='center', bbox=nodeType, arrowprops=arrow_args)


def plotMidText(counterPt, parentPt, textString):
    xMid = (parentPt[0] - counterPt[0]) / 2.0 + counterPt[0]
    yMid = (parentPt[1] - counterPt[1]) / 2.0 + counterPt[1]
    createPlot.ax1.text(xMid, yMid, textString)


def plotTree(myTree, parentPt, nodeText):
    numLeaves = getNumLeaves(myTree)
    treeDepth = getTreeDepth(myTree)
    firstString = myTree.keys()[0]
    counterPt = (plotTree.xOff + ((1.0 + float(numLeaves)) / 2.0) / plotTree.totalW, plotTree.yOff)
    plotMidText(counterPt, parentPt, nodeText)
    plotNode(firstString, counterPt, parentPt, decisionNode)
    secondDict = myTree[firstString]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], counterPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), counterPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), counterPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# def createPlot():
# fig=plt.figure(1,facecolor='white')
#     fig.clf()
#     createPlot.ax1=plt.subplot(111, frameon=False)
#     plotNode(' a decision node',(0.5,0.1),(0.1,0.5), decisionNode)
#     plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
#     plt.show()

def getNumLeaves(myTree):
    numLeaves = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeaves += getNumLeaves(secondDict[key])
        else:
            numLeaves += 1
    return numLeaves


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': \
                                                      {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': \
                                                      {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeaves(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


if __name__ == "__main__":
    myTree = retrieveTree(0)
    createPlot(myTree)