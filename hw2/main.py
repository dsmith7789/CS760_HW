import pandas as pd
import math

def gain_ratio(df: pd.DataFrame, split_feature: str, split: float) -> float:
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
    gain_ratio = (y_entropy - conditional_entropy) / split_entropy
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
    # we can get away with this because we're using binary splits
    for count in [class_0_count, class_1_count]:
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
    upper_split_df = df.loc[df[split_feature] >= split]
    lower_split_df = df.loc[df[split_feature] < split]
    conditional_entropy = 0
    upper_split_entropy = entropy(upper_split_df, feature="y", split=1)
    lower_split_entropy = entropy(lower_split_df, feature="y", split=1)
    conditional_entropy += (upper_split_entropy + lower_split_entropy)
    return -1 * conditional_entropy

def find_best_split(dataset: pd.DataFrame, candidate_splits: list[tuple[str, float]]) -> tuple[str, float]:
    '''Finds the split among the candidates that has the best gain ratio.

    Args:
        candidate_splits:
            A list of (str, float) candidate tuples that denotes a possible feature and value to split on.
    
    Returns:
        The candidate tuple that has the maximum gain ratio.
    '''
    best_gain_ratio = float("-inf")
    best_candidate = None
    for split_feature, split in candidate_splits:
        gain_ratio = gain_ratio(dataset, split_feature, split)
        
    pass

def stop_criteria_met():
    pass

def determine_split_candidates(instance_df: pd.DataFrame) -> list[tuple[str, float]]:
    '''Given a set of training instances, determine the feature/value pairs we can split on.

    '''
    split_candidates = []
    for feature in ["x1", "x2"]:
        print(f"this should be sorted ascending by {feature}:")
        instance_df.sort_values(by=feature, ignore_index=True, inplace=True)
        print(instance_df)
        for i in range(instance_df.shape[0] - 1):
            if instance_df.at[i, "y"] != instance_df.at[i + 1, "y"]:
                print(feature, instance_df.at[i, feature], instance_df.at[i, "y"])
                split_candidates.append((feature, instance_df.at[i, feature]))  # (feature, value) is a candidate split
    return split_candidates
    pass

def make_subtree(instance_df: pd.DataFrame):
    pass


def main():
    #s = determine_split_candidates(None)

    train_df = pd.read_csv("Homework_2_data/D1.txt", sep=" ", header=None, names=["x1", "x2", "y"])
    print(train_df.shape)
    split_candidates = determine_split_candidates(train_df)
    print(split_candidates)
    pass

if __name__ == "__main__":
    main()