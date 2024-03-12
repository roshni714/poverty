from opt_targeted_transfers import OptTargetedTransfers, ConditionalTargetedTransfers
from data_loaders import get_dataset
from data_utils import split_data


#Make train and test sets
X, y, r, features = get_dataset("malawi")
d = 3
(X_train, y_train, r_train), (X_test, y_test, r_test) = split_data(X=X, y=y, r=r, p=0.6)
X_train = X_train[:, :d]
X_test = X_test[:, :d]

tt = OptTargetedTransfers(name="malawi_example", c_bar=2.15, budget=None)

print("Fitting densities...")
# Fit density functions
tt.fit(X_train, y_train, r_train)

# Set budget (can do this at initialization or after fitting densities)
tt.set_budget(budget=0.1) 

# Run optimization algorithm
tt.run_opt(X_test, r_test, n_alpha=10, path="malawi_example_unconditional_budget=0.1.csv")

# Evaluate policy
res = tt.evaluate(X_test, y_test, r_test)
print(res)

# Query the optimal transfer policy after running the optimization algorithm
transfer = tt.opt_policy(X_test[[0]])
print(transfer)

# Can try a different budget without re-fitting the densities!
# Note that setting a new budget will clear the previously computed policy,
# so we will have to re-run the optimization.
tt.set_budget(budget=0.15)
tt.run_opt(X_test, r_test, n_alpha=10, path="malawi_example_unconditional_budget=0.15.csv")

# Evaluate policy
res = tt.evaluate(X_test, y_test, r_test)
print(res)
