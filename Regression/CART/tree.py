class treeNode(object):
    def __init__(self, feat=None, val=None, right=None, left=None):
        self.feature = feat
        self.value = val
        self.right = right
        self.left = left

    def __str__(self):
        return "feature : " + str(self.feature) + "\n" + \
               "value : " + str(self.value) + "\n" + \
               "right : " + str(self.right) + "\n" + \
               "left : " + str(self.left) + "\n"
