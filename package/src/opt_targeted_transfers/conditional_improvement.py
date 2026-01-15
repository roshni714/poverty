from opt_targeted_transfers.dataset_utils import standardize
import numpy as np
import torch
import tqdm
import copy


def get_avg_estimated_benefit(
    validation_dataset, loss_type, idx_to_receive_transfers, t, c_bar
):
    X, y, r = validation_dataset.get_data()

    if loss_type == "binary_gap":
        current_gaps = np.maximum(c_bar - y, 0)
        gaps_after_transfer = np.maximum(c_bar - t - y, 0)
        benefits = current_gaps - gaps_after_transfer
    elif loss_type == "binary_rate":
        current_gaps = (y <= c_bar).astype(float)
        gaps_after_transfer = (y + t <= c_bar).astype(float)
        benefits = current_gaps - gaps_after_transfer

    avg_estimated_benefits = np.sum(
        benefits[idx_to_receive_transfers] * r[idx_to_receive_transfers]
    )
    return avg_estimated_benefits


def get_conditional_improvement_loss(
    validation_dataset, loss_type, predictor, t, c_bar
):
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
    X, y, r = validation_dataset.get_data()
    if loss_type == "binary_gap":
        current_gaps = np.maximum(c_bar - y, 0)
        gaps_after_transfer = np.maximum(c_bar - t - y, 0)
        benefits = current_gaps - gaps_after_transfer
    elif loss_type == "binary_rate":
        current_gaps = (y <= c_bar).astype(float)
        gaps_after_transfer = (y + t <= c_bar).astype(float)
        benefits = current_gaps - gaps_after_transfer

    predicted_benefits = predictor(X)
    assert predicted_benefits.shape == benefits.shape
    mse_loss = (predicted_benefits - benefits) ** 2
    assert mse_loss.shape == r.shape
    weighted_mse_loss = np.sum(mse_loss * r) / np.sum(r)
    return weighted_mse_loss


def get_conditional_improvement_regressor(
    train_dataset,
    validation_dataset,
    loss_type,
    t,
    c_bar,
    n_layers=1,
    n_hidden_units=64,
    lr=5e-3,
    n_epochs=300,
    seed=123456,
    device="cpu",
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
    :param device: The device on which to train the neural network.
    :type device: str
    :return: The conditional gap improvement regressor
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    # shuffle the data
    X_train, y_train, r_train = train_dataset.get_data()
    X_val, y_val, r_val = validation_dataset.get_data()
    X_train, X_mean, X_std = standardize(X_train)
    X_val = (X_val - X_mean) / X_std

    if loss_type == "binary_gap":
        current_gaps_train = np.maximum(c_bar - y_train, 0)
        gaps_after_transfer_train = np.maximum(c_bar - t - y_train, 0)
        benefits_train = current_gaps_train - gaps_after_transfer_train

        current_gaps_val = np.maximum(c_bar - y_val, 0)
        gaps_after_transfer_val = np.maximum(c_bar - t - y_val, 0)
        benefits_val = current_gaps_val - gaps_after_transfer_val

    elif loss_type == "binary_rate":
        current_gaps_train = (y_train <= c_bar).astype(float)
        gaps_after_transfer_train = (y_train + t <= c_bar).astype(float)
        benefits_train = current_gaps_train - gaps_after_transfer_train

        current_gaps_val = (y_val <= c_bar).astype(float)
        gaps_after_transfer_val = (y_val + t <= c_bar).astype(float)
        benefits_val = current_gaps_val - gaps_after_transfer_val

    assert np.min(benefits_train) >= 0
    assert np.min(benefits_val) >= 0

    if X_train.shape[1] == 0:
        # TODO fill in
        pass
    else:
        d = X_train.shape[1]
        model_list = [torch.nn.Linear(d, n_hidden_units), torch.nn.ReLU()]
        for _ in range(n_layers - 1):
            model_list.append(torch.nn.Linear(n_hidden_units, n_hidden_units))
            model_list.append(torch.nn.ReLU())
        model_list.append(torch.nn.Linear(n_hidden_units, 1))
        predictor = torch.nn.Sequential(*model_list).to(device)

        def loss_function(predictor, X, benefits):
            predicted_benefits = predictor(torch.Tensor(X).to(device)).squeeze()
            actual_benefits = torch.Tensor(benefits).to(device)
            return (predicted_benefits - actual_benefits) ** 2

        optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

        batch_size = int(min(16000, len(X_train)) / 5)
        print(f"Fitting conditional {loss_type} improvement for transfer size {t}")
        pbar = tqdm.tqdm(list(range(n_epochs)))
        val_losses = []
        models = []

        for epoch in pbar:
            if epoch % 10 == 0:
                predictor.eval()
                val_loss = torch.sum(
                    loss_function(predictor, X_val, benefits_val)
                    * torch.Tensor(r_val).to(device)
                )
                val_losses.append(val_loss.detach().item())
                models.append(copy.deepcopy(predictor.cpu()))
            predictor = predictor.to(device)
            predictor.train()
            idx = np.random.choice(len(X_train), size=batch_size, replace=True)
            optimizer.zero_grad()

            unweighted_loss = loss_function(
                predictor, X_train[idx, :], benefits_train[idx]
            )
            weights = torch.Tensor(r_train[idx]).to(device)
            loss = torch.sum(unweighted_loss * weights)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"val loss": val_losses[-1]})

        best_model_idx = np.argmin(val_losses)
        final_predictor = models[best_model_idx]
        final_predictor.eval()
        final_predictor = final_predictor.cpu()

    def estimator(X_test):
        if X_test.shape[1] == 0:
            pass
        else:
            X_test = (X_test - X_mean) / X_std
            predicted_benefits = (
                (final_predictor(torch.Tensor(X_test)).reshape(X_test.shape[0], 1))
                .detach()
                .numpy()
                .flatten()
            )

        return np.maximum(predicted_benefits, 0)

    return estimator
