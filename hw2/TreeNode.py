class TreeNode:
    def __init__(self, feature_index: int, split: float, class_label: int) -> None:
        self.feature_index = feature_index
        self.split = split
        self.class_label = None # leave blank until we know it's a leaf node
        self.left = None
        self.right = None

    def set_class_label(self, class_label: int) -> None:
        ''' Marks node as a leaf, sets the class label associated with leaf node.

        Args:
            class_label:
                Either 0 or 1, since those are the only 2 labels available for this dataset.
        '''
        self.class_label = class_label
        pass