import pandas as pd

def gain_ratio():
    # y_entropy
    # y_conditional_entropy
    # split_entropy 
    pass

def entropy(feature: str) -> float:
    ''' Use standard formula to calculate entropy of dataset w.r.t some attribute.

    Entropy(feature) = -{sum[P(feature=val) * lg(P(feature=val))]} for all values of feature

    Args:
        feature:
            The column name (ex. "x1", "x2", "y")
    
    Returns:
        The entropy of the dataset with respect to that column
    '''
    pass

def conditional_entropy():
    pass

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
    for split in candidate_splits:
        gain_ratio = gain_ratio(dataset, split)
        
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