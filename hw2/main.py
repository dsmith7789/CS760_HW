import pandas as pd
import math
from TreeNode import TreeNode

# global to catch when stop criteria is met at any given point
stop_criteria_met = False

def calculate_gain_ratio(df: pd.DataFrame, split_feature: str, split: float) -> float:
    ''' Normalize information gain of a split by the entropy of that split.

    Calculates using standard formula of gain_ratio(d, s) = [H(Y) - H(Y|S)] / H(S)
    
    Args:
        df:
            The dataset we are calculating the entropy over.
        split_feature:
            The candidate split's feature (ex. split_feature = "x1" if considering a split of x1 >= 0.5)
        split:
            The value which if the feature is >= to, instances would go to one branch, but go to the other branch if < that split value.
            For example, split = 0.5 if considering a split of x1 >= 0.5.
    
    Returns:
        The result of the standard gain ratio formula shown above.
    '''
    y_entropy = entropy(df, feature="y", split=1)
    y_conditional_entropy = conditional_entropy(df, split_feature=split_feature, split=split)
    split_entropy = entropy(df, feature=split_feature, split=split)
    if split_entropy == 0:
        return 0
    gain_ratio = (y_entropy - y_conditional_entropy) / split_entropy
    return gain_ratio

def entropy(df:pd.DataFrame, feature: str, split: float) -> float:
    ''' Use standard formula to calculate entropy of dataset w.r.t some attribute.

    Entropy(feature) = -{sum[P(feature=val) * lg(P(feature=val))]} for all values of feature

    Args:
        df:
            The dataset we are calculating the column's entropy for.
        feature:
            The column name (ex. "x1", "x2", "y") within the dataset.
        split:
            If a value is greater than or equal to the split, it goes in one branch; otherwise it goes in the other branch.    
    Returns:
        The entropy of the dataset with respect to that column
    '''
    entropy = 0
    class_0_count = 0
    class_1_count = 0
    total = df.shape[0]
    for i in df.index:
        if df[feature][i] < split:
            class_0_count += 1
        else:
            class_1_count += 1

    if class_0_count == 0 or class_1_count == 0:
        return 0
    
    # we can get away with this because we're using binary splits
    for count in [class_0_count, class_1_count]:
        # print(f"count = {count}")
        entropy += (count / total) * math.log((count / total), 2)
    return -1 * entropy

def conditional_entropy(df: pd.DataFrame, split_feature: str, split: float):
    ''' Calculates the conditional entropy of the output with respect to a feature and a condition.

    For example, if trying to calculate H(Y|X2 >= 0.5), you would put in:
    conditional_entropy(df, condition_var="x2", condition=0.5)

    Args:
        df:
            The dataset we are calculating entropy over.
        condition_var:
            The feature we have a condition on (ex. what is the conditional entropy of Y if we know feature >= 0.5?)
        condition:
            The value we are splitting the feature values on. Since this is a binary split, we can consider the branches
            where feature < condition and where feature >= condition.
    '''
    upper_split_df = df[df[split_feature] >= split]
    lower_split_df = df[df[split_feature] < split]
    conditional_entropy = 0
    upper_split_entropy = entropy(upper_split_df, feature="y", split=1)
    lower_split_entropy = entropy(lower_split_df, feature="y", split=1)
    conditional_entropy += (upper_split_entropy + lower_split_entropy)
    return conditional_entropy

def find_best_split(dataset: pd.DataFrame, candidate_splits: list[tuple[str, float]]) -> tuple[str, float]:
    '''Finds the split among the candidates that has the best gain ratio.

    Args:
        candidate_splits:
            A list of (str, float) candidate tuples that denotes a possible feature and value to split on.
    
    Returns:
        The candidate tuple that has the maximum gain ratio.
    '''
    best_gain_ratio = 0
    best_candidate = (None, None)
    global stop_criteria_met
    for split_feature, split in candidate_splits:
        gain_ratio = calculate_gain_ratio(dataset, split_feature, split)
        if gain_ratio >= best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_candidate = (split_feature, split)
    if best_gain_ratio == 0:
        stop_criteria_met = True
    return best_candidate

def majority_class(df: pd.DataFrame) -> int:
    ''' Finds the majority output class among instances in a dataset.

    Used to determine what the class label is when making a leaf node in the tree.

    Args:
        df:
            The dataset we are considering for this leaf node.
    
    Returns:
        0 or 1, depending on which output class is the majority (returns 1 in the case of a tie).
    '''
    class_0_count = 0
    class_1_count = 1
    for i in df.index:
        if df["y"][i] == 0:
            class_0_count += 1
        else:
            class_1_count += 1
    
    return 0 if class_0_count > class_1_count else 1

def determine_split_candidates(df: pd.DataFrame) -> list[tuple[str, float]]:
    '''Given a set of training instances, determine the feature/value pairs we can split on.

    Args:
        df:
            The dataset we are trying to split.
    Returns:
        A list of (feature, value) tuples that we can split on (ex. [("x1", 0.5), ("x2", 0.6)])
    '''
    split_candidates = []
    for feature in ["x1", "x2"]:
        df.sort_values(by=feature, ignore_index=True, inplace=True)
        for i in range(df.shape[0] - 1):
            if df.at[i, "y"] != df.at[i + 1, "y"]:
                split_candidates.append((feature, df.at[i + 1, feature]))  # (feature, value) is a candidate split
    return split_candidates

def make_subtree(df: pd.DataFrame) -> TreeNode:
    print(f"dataset size = {df.shape}")
    global stop_criteria_met
    node = TreeNode()
    best_split_feature, best_split_value = None, None # set later

    if not df.empty:
        split_candidates = determine_split_candidates(df)
        (best_split_feature, best_split_value) = find_best_split(df, split_candidates)
    else:
        stop_criteria_met = True

    if stop_criteria_met:
        # make leaf node
        node.leaf = True
        node.class_label = majority_class(df)
        stop_criteria_met = False
    else:
        # make internal node
        node.feature = best_split_feature
        node.split_value = best_split_value

        left_df = df[df[best_split_feature] < best_split_value]
        print("calculating left subtree")
        left_subtree = make_subtree(left_df)
        right_df = df[df[best_split_feature] >= best_split_value]
        print("calculating right subtree")
        right_subtree = make_subtree(right_df)

        node.left = left_subtree
        node.right = right_subtree
    
    return node


def main():
    #s = determine_split_candidates(None)

    df = pd.read_csv("Homework_2_data/D1.txt", sep=" ", header=None, names=["x1", "x2", "y"])
    root = make_subtree(df)
    print(root)

if __name__ == "__main__":
    main()