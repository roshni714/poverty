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

Poverty rate targeting
```
from opt_targeted_transfers import UnconditionalTargetedTransfers 
from examples.data_loaders import get_example_data

train_dataset, test_dataset = get_example_data()

tt = RateTargetedTransfers(c_bar=2.15, budget=None)
tt.fit(train_dataset)
tt.set_budget(0.5)

tt.run_opt(
   test_covariate_dataset, n_alpha=100, path="malawi_example_rate_B=0.5.csv"
)

res = tt.evaluate(test_dataset)
```
