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

Load simple dataset.
```
from opt_targeted_transfers import RateTargetedTransfers
from opt_targeted_transfers import Dataset
from data_loaders import load_data, PATH_TO_TRAIN_DATA, PATH_TO_TEST_DATA

train_data = load_data(PATH_TO_TRAIN_DATA)
test_data = load_data(PATH_TO_TEST_DATA)

train_dataset = Dataset(df=train_data, outcome='consumption_per_capita_per_day', weight='hh_wgt', covs=['hh_size', 'urban'])
test_covariate_dataset = Dataset(df=test_data, outcome=None, weight='hh_wgt', covs=['hh_size', 'urban'])
test_dataset = Dataset(df=test_data, outcome='consumption_per_capita_per_day', weight='hh_wgt', covs=['hh_size', 'urban'])

```

Poverty rate targeting.
```
tt = RateTargetedTransfers(c_bar=2.15, budget=None)
tt.fit(train_dataset)
tt.set_budget(0.5)

tt.run_opt(
   test_covariate_dataset, n_alpha=100, path="malawi_example_rate_B=0.5.csv"
)

res = tt.evaluate(test_dataset)
```
