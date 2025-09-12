from sklearn.linear_model import Ridge
import numpy as np
from opt_targeted_transfers import standardize


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

    ordered_features = []
    feature_list = sorted(train_dataset.covs.copy())

    scores = []

    for i in range(min(max_features, len(feature_list))):
        best_score = 0
        best_feature = None
        best_model = None

        for feature in feature_list:
            features = ordered_features + [feature]
            train_dataset.covs = features
            X, y, r = train_dataset.get_data()
            X, X_mean, X_std = standardize(X)
            y = np.clip(y, None, truncation_upper_value)
            y, y_mean, y_std = standardize(y)

            if X.shape[0] > 16000:
                np.random.seed(3758926)
                sample_indices = np.random.choice(X.shape[0], size=16000, replace=False)
                X = X[sample_indices]
                y = y[sample_indices]
                r = r[sample_indices]

            model = Ridge(fit_intercept=True)
            model.fit(X, y, sample_weight=r)

            validation_dataset.covs = features
            X_val, y_val, r_val = validation_dataset.get_data()
            X_val = (X_val - X_mean) / X_std
            y_val = np.clip(y_val, None, truncation_upper_value)
            y_val = (y_val - y_mean) / y_std

            score = model.score(X_val, y_val, sample_weight=r_val)

            if score > best_score:
                best_score = score
                best_feature = feature
                best_model = model
        ordered_features.append(best_feature)
        scores.append(best_score)
        feature_list.remove(best_feature)
        print(best_feature, best_score)

    return ordered_features, scores
