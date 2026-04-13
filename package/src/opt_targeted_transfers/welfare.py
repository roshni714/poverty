from opt_targeted_transfers.opt import TargetedTransfers
import numpy as np
from opt_targeted_transfers.dataset_utils import standardize
from opt_targeted_transfers.evaluate import policy_cost
import numpy as np
import torch
import tqdm
import copy


def get_conditional_marginal_utility_estimator(
    train_dataset,
    validation_dataset,
    utility_derivative_func,
    t,
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
    
    benefits_train = utility_derivative_func(y_train + t)
    benefits_val = utility_derivative_func(y_val + t)

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
        print(f"Fitting conditional marginal utility for transfer size {t}")
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

class WelfareTargetedTransfers(TargetedTransfers):

    def __init__(self, c_bar, budget):
        super().__init__(c_bar=c_bar, budget=budget)
        self.utility_func = lambda x: np.log(x)
        self.utility_derivative_func = lambda x: 1/x
        self.budget = budget
        self.name = "welfare"

        self.n_regressors = 20
        self.candidate_t_values = np.linspace(0.01, 10, self.n_regressors)
    
    def _get_assignments_for_lambda(self, X, transfer_marginal_utility_grid, lambda_):

        # transfer_marginal_utility_grid is num_transfer_sizes x n
        # for each household, find the transfer size with the largest marginal utility below lambda
        grid_less_than_lambda = transfer_marginal_utility_grid <= lambda_
        # get index of largest transfer size with marginal utility above lambda
        idx = np.maximum(np.sum(grid_less_than_lambda, axis=1) - 1, 0)
        assert np.all(idx >= 0)
        assignments = {i: [(self.candidate_t_values[idx[i]], 1.0)] for i in range(X.shape[0])}
        return assignments



    def fit(self,
        train_dataset,
        validation_dataset,
        n_layers=1,
        n_hidden_units=256,
        lr=5e-3,
        n_epochs=300,
        seed=123456,
        device="cpu",
    ):

        # For each transfer size t, fit conditional marginal utility estimator
        self.t_to_household_estimator_map = dict()
        for transfer_size in self.candidate_t_values:
            self.t_to_household_estimator_map[transfer_size] = get_conditional_marginal_utility_estimator(
                train_dataset,
                validation_dataset=validation_dataset,
                t=transfer_size,
                utility_derivative_func=self.utility_derivative_func,
                n_layers=n_layers,
                n_hidden_units=n_hidden_units,
                lr=lr,
                n_epochs=n_epochs,
                seed=seed,
                device=device,
            )
    
    def run_opt(self, test_covariate_dataset):
        X_test, _ = test_covariate_dataset.get_data()

        res = []
        lambda_min = float("inf")
        lambda_max = 0
        for transfer_size in self.candidate_t_values:
            household_estimator = self.t_to_household_estimator_map[transfer_size]
            predicted_benefits = household_estimator(X_test)
            lambda_min = min(lambda_min, np.min(predicted_benefits))
            lambda_max = max(lambda_max, np.max(predicted_benefits))
            res.append(predicted_benefits)
        transfer_marginal_utility_grid = np.array(res).T # num_transfer_sizes x n_train
        
        lambda_mid = (lambda_min + lambda_max) / 2
        assignments = self._get_assignments_for_lambda(X_test, transfer_marginal_utility_grid, lambda_mid)
        lamb_cost = policy_cost(test_covariate_dataset, assignments)

        low = lambda_min
        high = lambda_max
        lambda_value = lambda_mid

        while np.abs(lamb_cost - self.budget) > 1e-3:
            if lamb_cost > self.budget:
                high = lambda_value
            else:
                low = lambda_value
            next_lambda_value = (high + low) / 2
            assignments = self._get_assignments_for_lambda(X_test, transfer_marginal_utility_grid, next_lambda_value)
            next_lamb_cost = policy_cost(test_covariate_dataset, assignments)
            lambda_value = next_lambda_value
            lamb_cost = next_lamb_cost
            print(f"lambda: {lambda_value}, cost: {lamb_cost}", high, low)
        
        self.assignments = assignments
        return assignments


        

        
        



        


        
        
        



    
