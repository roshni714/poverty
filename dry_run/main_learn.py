import yaml
import argh
import numpy as np
from feature_selection import forward_selection
from opt_targeted_transfers import RateTargetedTransfers, GapTargetedTransfers, BinaryRateTargetedTransfers, BinaryGapTargetedTransfers, write_result

C_BAR = 2.15
BUDGETS=np.linspace(0.05, 2.15, 30)

def learn_continuous_rate(train_dataset, validation_dataset, test_covariate_dataset, test_dataset, continuous_rate_params, savepath):
    """
    Learn the continuous rate targeted transfers
    """
    features, _ = forward_selection(train_dataset, validation_dataset, max_features=continuous_rate_params["density_estimation"]["n_features"])

    tt = RateTargetedTransfers(c_bar=C_BAR)
    tt.fit(train_dataset, n_knots=continuous_rate_params["density_estimation"]["n_knots"], 
           n_quantiles=continuous_rate_params["density_estimation"]["n_bins"], 
           degree=continuous_rate_params["density_estimation"]["degree"])
    for budget in BUDGETS:
        tt.set_budget(budget)
        tt.run_opt(test_covariate_dataset, continuous_rate_params["n_alpha"])
        res = tt.evaluate(test_dataset)
        write_result(savepath + ".csv", res)
    auc_res = tt.evaluate_auc(test_dataset)



@argh.arg("--config", default="hparam_results/output_gan_binary_gap.yaml")
@argh.arg("--trainpath", default="data/train.parquet")
@argh.arg("--testpath", default="data/test.parquet")
@argh.arg("--savedir", default="learn_results")
def main(config="hparam_results/output_gan_binary_gap.yaml", trainpath="data/train.parquet", testpath="data/test.parquet", savedir="learn_results"):
    """
    Main function to learn and evaluate targeted transfers.
    """
    with open(config) as stream:
        try:
            config_hparam = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    config_keys = list(config_hparam.keys())
    assert all([key in ['continuous_rate', 'binary_rate', 'continuous_gap', 'binary_gap'] for key in config_keys])

    train_dataset = load_data(trainpath, testpath)
    test_dataset = load_data(testpath, trainpath)

    name = config.split("/")[1].split(".yaml")[0]
    savepath = savedir + "/" + name 

    for key in config_keys:
        if key == 'continuous_rate':
            continuous_rate_params = config[key]
        elif key == 'binary_rate':
            binary_rate_params = config[key]
        elif key == 'continuous_gap':
            continuous_gap_params = config[key]
        elif key == 'binary_gap':
            binary_gap_params = config[key]
