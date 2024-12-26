from sklearn.linear_model import LinearRegression
import numpy as np


def forward_selection(
    train_dataset, validation_dataset, truncation_upper_value=10, max_features=50
):
    """
    Perform forward selection.

    Args:
        train_dataset (Dataset): The training dataset.
        validation_dataset (Dataset): The validation dataset.
        truncation_upper_value (float): The outcome space is truncated at this value.

    Returns:
        ordered_features (list): A list of features in the order they were selected.
        scores (list): A list of R^2 scores for each feature set.
    """

    # TODO: Should we clip the outcomes to the truncation_upper_value? Or should we remove these samples?

    feature_list = train_dataset.df.columns.tolist()
    outcome = train_dataset.outcome
    weight = train_dataset.weight
    feature_list.remove(outcome)
    feature_list.remove(weight)
    ordered_features = []

    filtered_train_df = train_dataset.df[
        train_dataset.df[outcome] < truncation_upper_value
    ]
    filtered_validation_df = validation_dataset.df[
        validation_dataset.df[outcome] < truncation_upper_value
    ]

    y_val = filtered_validation_df[outcome].values
    r_val = filtered_validation_df[weight].values
    scores = []

    for i in range(max_features):
        best_score = 0
        best_feature = None

        for feature in feature_list:
            features = ordered_features + [feature]
            y = filtered_train_df[outcome].values
            X = filtered_train_df[features].values
            r = filtered_train_df[weight].values
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y, sample_weight=r)
            X_val = filtered_validation_df[features].values
            score = model.score(X_val, y_val, sample_weight=r_val)

            if score > best_score:
                best_score = score
                best_feature = feature

        ordered_features.append(best_feature)
        scores.append(best_score)
        feature_list.remove(best_feature)
        print(best_feature, best_score)

    return ordered_features, scores
