import statsmodels.api as sm
import numpy as np

def forward_selection(train_dataset, validation_dataset):
    """
    Perform forward selection.

    Args:
        train_dataset (pd.DataFrame): The training dataset.
        validation_dataset (pd.DataFrame): The validation dataset.
    
    Returns:
        ordered_features (list): A list of features in the order they were selected.
        scores (list): A list of R^2 scores for each feature set.
    """

    feature_list = train_dataset.columns.tolist()
    feature_list.remove("consumption_per_capita_per_day")
    feature_list.remove("hh_wgt")
    ordered_features = []

    y_val = validation_dataset["consumption_per_capita_per_day"].flatten()
    scores = []

    for i in range(len(feature_list)):
        best_score = 0
        best_feature = None

        for feature in feature_list:
            features = ordered_features + [feature]
            y = train_dataset["consumption_per_capita_per_day"].to_numpy()
            X = train_dataset[features].to_numpy()
            model = sm.OLS(y, X, hasconst=True)
            results = model.fit()
            y_hat = results.predict(validation_dataset[features].to_numpy())
            score = (np.corrcoef(y_val, y_hat.flatten()).item()) ** 2

            if score > best_score:
                best_score = score
                best_feature = feature

        ordered_features.append(best_feature)
        scores.append(best_score)
        feature_list.remove(best_feature)
    
    return ordered_features, scores