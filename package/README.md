# opt_targeted_transfers

This package implements methods for learning targeted transfer policies.

## Installation
Here are the installation instructions.
```
pip install virtualenv
python3 -m venv opt
source opt/bin/activate
cd poverty/package
pip install -r requirements.txt
pip install -e .
```

## Usage

Learn the optimal (unconditional) targeted transfer policy.
```
from opt_targeted_transfers import OptTargetedTransfers 
from examples.data_loaders import get_dataset

X_train, y_train, X_test, y_test = get_dataset("simple")

tt = OptTargetedTransfers(name="malawi_example", c_bar=2.15, budget=None)
tt.fit(X_train, y_train)

tt.set_budget(budget=0.1)
tt.run_opt(X_test, n_alpha=10)
res = tt.evaluate(X_test, y_test)
transfer = tt.opt_policy(X_test[[0]])

tt.set_budget(budget=0.15)
tt.run_opt(X_test, n_alpha=10)
res = tt.evaluate(X_test, y_test)
```

Learn the optimal conditional targeted transfer policy.

```
```
