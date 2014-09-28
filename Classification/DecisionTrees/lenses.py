from treePlotter import *
from trees import *


def main():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    createPlot(lensesTree)


if __name__ == "__main__":
    main()

