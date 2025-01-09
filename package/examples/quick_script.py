from opt_targeted_transfers import RateTargetedTransfers
from opt_targeted_transfers import Dataset
from data_loaders import load_data, PATH_TO_TRAIN_DATA, PATH_TO_TEST_DATA


# Make train and test set
train_data = load_data(PATH_TO_TRAIN_DATA)
test_data = load_data(PATH_TO_TEST_DATA)

train_dataset = Dataset(df=train_data, outcome='consumption_per_capita_per_day', weight='hh_wgt', covs=['hh_size', 'urban'])
test_covariate_dataset = Dataset(df=test_data, outcome=None, weight='hh_wgt', covs=['hh_size', 'urban'])
test_dataset = Dataset(df=test_data, outcome='consumption_per_capita_per_day', weight='hh_wgt', covs=['hh_size', 'urban'])


tt = RateTargetedTransfers(c_bar=2.15, budget=None)
tt.fit(train_dataset)
tt.set_budget(0.5)

tt.run_opt(
   test_covariate_dataset, n_alpha=1, max_alpha=0.2, min_alpha=0.2, path="malawi_example_rate_B=0.5.csv"
)

res = tt.evaluate(test_dataset)
print(res)