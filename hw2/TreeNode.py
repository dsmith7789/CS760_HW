class TreeNode:
    def __init__(self) -> None:
        ''' Representation of a node in the decision tree.
        '''
        self.feature = None  # The feature we would be splitting on (ex. "x1", "x2")
        self.split_value = None  # The value of the feature that we are splitting on (ex. x1 >= 0.5, then split = 0.5)
        self.leaf = False   # only need to set if we're at a leaf node
        self.class_label = None # leave blank until we know it's a leaf node
        self.left = None
        self.right = None