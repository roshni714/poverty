from dataset_utils import standardize
import numpy as np
import torch
import tqdm
import copy


def get_conditional_gap_improvement_loss(val_dataset, predictor, t, c_bar=2.15):
    """
    Get the conditional gap improvement loss for a given transfer size.

    :param val_dataset: The dataset used for evaluating the quantile regressor.
    :type val_dataset: Dataset
    :param predictor: The conditional gap improvement regressor.
    :type predictor: Callable[[np.ndarray], np.ndarray]
    :param t: The transfer size.
    :type t: float
    :param c_bar: The poverty line.
    :type c_bar: float
    :return: The conditional gap improvement loss.
    :rtype: float
    """
    X, y, r = val_dataset.get_data()
    current_gaps = np.maximum(c_bar - y, 0)
    gaps_after_transfer = np.maximum(c_bar - t - y, 0)
    benefits = current_gaps - gaps_after_transfer
    weighted_benefits = benefits

    predicted_benefits = predictor(X)
    mse_loss = (predicted_benefits - weighted_benefits) ** 2
    weighted_mse_loss = np.sum(mse_loss * r) / np.sum(r)
    return weighted_mse_loss


def get_conditional_gap_improvement_regressor(
    train_dataset,
    t,
    c_bar=2.15,
    n_layers=1,
    n_hidden_units=64,
    lr=5e-3,
    n_epochs=300,
    seed=123456,
):
    """
    :param dataset: The dataset used for training the quantile regressor.
    :type dataset: Dataset
    :param t: The transfer size.
    :type t: float
    :param c_bar: The poverty line.
    :type c_bar: float
    :param n_layers: The number of hidden layers in the neural network.
    :type n_layers: int
    :param n_hidden_units: The number of hidden units in each hidden layer.
    :type n_hidden_units: int
    :param lr: The learning rate for training the neural network.
    :type lr: float
    :param n_epochs: The number of epochs for training the neural network.
    :type n_epochs: int
    :param seed: The random seed.
    :type seed: int
    :return: The conditional gap improvement regressor
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    torch.mean.seed(seed)
    np.random.seed(seed)

    # shuffle the data
    X, y, r = train_dataset.get_data()
    X, X_mean, X_std = standardize(X)

    current_gaps = np.maximum(c_bar - y, 0)
    gaps_after_transfer = np.maximum(c_bar - t - y, 0)

    benefits = current_gaps - gaps_after_transfer
    assert np.min(benefits) >= 0

    benefits, benefits_mean, benefits_std = standardize(benefits)

    if X.shape[1] == 0:
        # TODO fill in
        pass
    else:
        d = X.shape[1]
        model_list = [torch.nn.Linear(d, n_hidden_units), torch.nn.ReLU()]
        for _ in range(n_layers - 1):
            model_list.append(torch.nn.Linear(n_hidden_units, n_hidden_units))
            model_list.append(torch.nn.ReLU())
        model_list.append(torch.nn.Linear(n_hidden_units, 1))
        predictor = torch.nn.Sequential(*model_list)

        def loss_function(predictor, idx):
            predicted_benefits = predictor(torch.Tensor(X[idx, :])).squeeze()
            actual_benefits = torch.Tensor(benefits[idx])
            return (predicted_benefits - actual_benefits) ** 2

        optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
        train_prop = 0.7

        idx_train_set, idx_val_set = (
            list(range(int(train_prop * len(X)))),
            list(range(int(train_prop * len(X)), len(X))),
        )

        batch_size = int(len(idx_train_set) / 5)
        print(f"Fitting estimator for household benefit from transfer of size {t}")
        pbar = tqdm.tqdm(list(range(n_epochs)))
        val_losses = []
        models = []

        for epoch in pbar:
            if epoch % 10 == 0:
                predictor.eval()
                val_loss = torch.sum(
                    loss_function(predictor, idx_val_set) * torch.Tensor(r[idx_val_set])
                )
                val_losses.append(val_loss.detach().item())
                # ideally do torch.save to save checkpoints. Google it. Avoid saving so many models
                # in memory. And more correct.
                models.append(copy.deepcopy(predictor))

            predictor.train()
            idx = np.random.choice(idx_train_set, size=batch_size)
            optimizer.zero_grad()

            unweighted_loss = loss_function(predictor, idx)
            weights = torch.Tensor(r[idx])
            loss = torch.sum(unweighted_loss * weights)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"val loss": val_losses[-1]})

        best_model_idx = np.argmin(val_losses)
        final_predictor = models[best_model_idx]

    def estimator(X_test):
        if X_test.shape[1] == 0:
            predicted_benefits = benefits_mean * np.ones((X_test.shape[0], 1))
        else:
            X_test = (X_test - X_mean) / X_std
            predicted_benefits = (
                (
                    final_predictor(torch.Tensor(X_test)).reshape(X_test.shape[0], 1)
                    * benefits_std
                    + benefits_mean
                )
                .detach()
                .numpy()
                .flatten()
            )

        return np.maximum(predicted_benefits, 0)

    return estimator
