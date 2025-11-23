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
from opt_targeted_transfers import Dataset
from data_loaders import load_data, PATH_TO_TRAIN_DATA, PATH_TO_TEST_DATA

train_data = load_data(PATH_TO_TRAIN_DATA)
test_data = load_data(PATH_TO_TEST_DATA)

train_dataset = Dataset(df=train_data, outcome='consumption_per_capita_per_day', weight='hh_wgt', covs=['hh_size', 'urban'])
train_dataset, validation_dataset = split(train_dataset)
test_covariate_dataset = Dataset(df=test_data, outcome=None, weight='hh_wgt', covs=['hh_size', 'urban'])
test_dataset = Dataset(df=test_data, outcome='consumption_per_capita_per_day', weight='hh_wgt', covs=['hh_size', 'urban'])

```

Poverty rate targeting.
```
from opt_targeted_transfers import RateTargetedTransfers

tt = RateTargetedTransfers(c_bar=CBAR, budget=None)
tt.fit(train_dataset, validation_dataset)
tt.set_budget(0.5)
tt.run_opt(
   test_covariate_dataset, n_alpha=100, path="malawi_example_rate_B=0.5.csv"
)
res = tt.evaluate(test_dataset)
auc_res = tt.compute_auc(test_covariate_dataset=test_covariate_dataset, test_dataset=test_dataset, metrics=["post_transfer_poverty_rate",
                                                              "post_transfer_poverty_gap"], budgets=[0.05, 0.1, 0.5, 1.0, 2.0])
```

Poverty gap targeting
```
from opt_targeted_transfers import GapTargetedTransfers

tt = GapTargetedTransfers(c_bar=CBAR, budget=None)
tt.fit(train_dataset, validation_dataset)
tt.set_budget(0.5)
tt.run_opt(test_covariate_dataset)
res = tt.evaluate(test_dataset)
```

Binary gap targeting
```
from opt_targeted_transfers import BinaryGapTargetedTransfers

tt = BinaryGapTargetedTransfers(c_bar=CBAR, n_transfer_values=5)
tt.fit(train_dataset, validation_dataset)
tt.get_opt_transfer_sizes_given_budget_grid(validation_dataset, budgets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, CBAR])
tt.set_budget(0.5)
tt.run_opt(test_covariate_dataset)
res = tt.evaluate(test_dataset)
auc_res = tt.compute_auc(test_covariate_dataset=test_covariate_dataset, test_dataset=test_dataset, metrics=["post_transfer_poverty_rate",
                                                              "post_transfer_poverty_gap"], budgets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, CBAR])
```

Binary rate targeting
```
from opt_targeted_transfers import BinaryRateTargetedTransfers

tt = BinaryGapTargetedTransfers(c_bar=CBAR, n_transfer_values=5)
tt.fit(train_dataset, validation_dataset)
tt.get_opt_transfer_sizes_given_budget_grid(validation_dataset, budgets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, CBAR])
tt.set_budget(0.5)
tt.run_opt(test_covariate_dataset)
res = tt.evaluate(test_dataset)
auc_res = tt.compute_auc(test_covariate_dataset=test_covariate_dataset, test_dataset=test_dataset, metrics=["post_transfer_poverty_rate",
                                                              "post_transfer_poverty_gap"], budgets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, CBAR])
```

Oracle rate and gap targeting
```
from opt_targeted_transfers import OracleRateTargetedTransfers, OracleGapTargetedTransfers

tt = OracleGapTargetedTransfers(c_bar=CBAR, budget=0.25)
tt.run_opt(test_dataset)
res = tt.evaluate(test_dataset)
auc_res = tt.compute_auc(test_dataset=test_dataset, metrics=["post_transfer_poverty_rate",
                                                              "post_transfer_poverty_gap"], budgets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, CBAR])

tt = OracleGapTargetedTransfers(c_bar=CBAR, budget=0.25, scheme="floor")
tt.run_opt(test_dataset)
res = tt.evaluate(test_dataset)
auc_res = tt.compute_auc(test_dataset=test_dataset, metrics=["post_transfer_poverty_rate",
                                                              "post_transfer_poverty_gap"], budgets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, CBAR])

```



