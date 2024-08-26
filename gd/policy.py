from opt_targeted_transfers import (
    ConditionalTargetedTransfers,
    UnconditionalTargetedTransfers,
    BinaryTargetedTransfers,
    OraclePovertyRateTargetedTransfers,
    BinaryConditionalTargetedTransfers
)
from data_loaders import get_datasets, get_district_dataset
from data_utils import aggregate_metrics

CBAR = 2.15

COVARIATE_LIST = [
    "num_children", 
    "durable_asset_Bed", 
    "hh_f12",
    "hh_f34", #num cellphones
    "hhsize", #household size
    "hh_f06", #construction materials categories
    "ag_asset_AXE",
    "durable_asset_Television",
    "hh_x07", #own any livestock?
    "durable_asset_Radio with flash drive/micro CD",
    "durable_asset_Motorcycle / Scooter",
    "ag_asset_WATERING CAN",
    "ag_asset_PANGA KNIFE",
    "popdensity",
    "district",
    "hh_f11", #lighting fuel categories
    "hh_f43", #rubbish disposal
    "hh_x04",
    "hh_f41", #toilet categories
    "durable_asset_Bicycle",
    "ag_asset_CHICKEN HOUSE",
    "hh_f08", #roof materials categories
    "hh_f01",
    "ag_asset_HAND HOE",
    "durable_asset_Refrigerator",
    "durable_asset_Car",
    "durable_asset_Radio ('wireless')",
    "durable_asset_Iron (for pressing clothes)",
    "ag_asset_GRANARY",
    "ag_asset_SLASHER"
    ]

RURAL_COVARIATE_LIST = [
    "hh_h03b", #meals per day
    "hh_t10", #what does HH head sleep on
    "hhsize", #household size
    "hh_f35_3", #how much money of cellphone cost spent on airtime usage
    "district",
    "durable_asset_Television",
    'hh_f06',
    "ag_asset_PANGA KNIFE",
    "hh_f12",
    ] 


def get_covariates(district, numfeatures):
    if district == "all":
        return COVARIATE_LIST[:numfeatures]
    else:
        return RURAL_COVARIATE_LIST[:numfeatures]
    


def saturation_policy(district, uncondtol, pool, numfeatures=None):
    fold1, fold2, features = get_datasets(district, pool, covariates=None)

    def run(fold_fit, fold_opt):
        tt = ConditionalTargetedTransfers(
            method="density", c_bar=CBAR, conditional_tolerance=uncondtol
        )
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, n_epochs=500, internal_knots=[min(y_fit), 1.0, 
                                                           2.0, 5, max(y_fit)])
        tt.run_opt(X_opt)
        metrics = tt.evaluate(X_opt, y_opt)
        tt.evaluate_equity(
            X_opt,
            y_opt,
            path="results/"
            + "{}_equity_{}_uncondtol={}.csv".format(district, "saturation", uncondtol),
        )
        tt.save_opt_policy(
            "policies/{}_saturation_uncondtol={}".format(district, uncondtol)
        )
        return metrics

    metrics1 = run(fold1, fold2)
    final_metrics = get_final_metrics(metrics1)
    final_metrics["n_opt"] = len(fold2[0])
    final_metrics["policy"] = "saturation"

    final_metrics["numfeatures"] = 0
    return final_metrics


def geographic_policy(district, uncondtol, numfeatures=None, pool=None):
    fold1, fold2, features = get_datasets(
        district,
        pool,
        covariates=[
            "district",
        ],
    )

    def run(fold_fit, fold_opt):
        tt = UnconditionalTargetedTransfers(
            c_bar=CBAR, unconditional_tolerance=uncondtol
        )
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, n_epochs=500, internal_knots=[min(y_fit), 1.0,
                                                           2.0, 5, max(y_fit)])
        tt.run_opt(
            X_opt,
            path="results/{}_{}_uncondtol={}_opt.csv".format(
                district, "geographic", uncondtol
            ),
        )
        metrics = tt.evaluate(X_opt, y_opt)
        tt.evaluate_equity(
            X_opt,
            y_opt,
            path="results/"
            + "{}_equity_{}_uncondtol={}.csv".format(district, "geographic", uncondtol),
        )
        tt.save_opt_policy(
            "policies/{}_geographic_uncondtol={}".format(district, uncondtol)
        )
        return metrics

    metrics1 = run(fold1, fold2)
    metrics2 = run(fold2, fold1)
    final_metrics = aggregate_metrics(metrics1, metrics2)
    final_metrics["n_opt"] = len(fold2[0])
    final_metrics["policy"] = "geographic"
    final_metrics["numfeatures"] = 1

    return final_metrics


def binary_targeting_policy(district, uncondtol, pool, numfeatures=None):

    # "hh_t10" - what does head of house sleep on (bed/mattress)
    # "hh_f12" - what is your main source of cooking fuel
    # "hh_f41" - what kind of toilet
    # "hh_t03" - sufficient amount of clothing
    # "hh_t19" - did you not eat bc no food
    # "hh_t04" - concerning the standard of health care you received

    fold1, fold2, features = get_datasets(
        district, pool, covariates=get_covariates(district, numfeatures=numfeatures)
    )

    def run(fold_fit, fold_opt):
        tt = BinaryTargetedTransfers(c_bar=CBAR, unconditional_tolerance=uncondtol)
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, n_epochs=500, internal_knots=[min(y_fit), 1.0, 
                                                           2.0, 5, max(y_fit)])
        tt.run_opt(X_opt)
        metrics = tt.evaluate(X_opt, y_opt)
        tt.evaluate_equity(
            X_opt,
            y_opt,
            path="results/"
            + "{}_equity_{}_uncondtol={}_numfeatures={}.csv".format(district, "binary", uncondtol, numfeatures),
        )
        tt.save_opt_policy(
            "policies/{}_binary_uncondtol={}".format(district, uncondtol)
        )
        return metrics

    metrics1 = run(fold1, fold2)
    final_metrics = get_final_metrics(metrics1)
    final_metrics["n_opt"] = len(fold2[0])
    final_metrics["policy"] = "binary"
    final_metrics["numfeatures"] = numfeatures

    return final_metrics


def optimized_policy(district, uncondtol, pool, numfeatures=None):

    fold1, fold2, features = get_datasets(
        district, pool, covariates=get_covariates(district, numfeatures)
    )

    def run(fold_fit, fold_opt):
        tt = UnconditionalTargetedTransfers(
            c_bar=CBAR, unconditional_tolerance=uncondtol
        )
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, n_epochs=500)
        tt.run_opt(
            X_opt,
            path="results/{}_{}_uncondtol={}_opt.csv".format(
                district, "optimized", uncondtol, n_alpha=300
            ),
            
        )
        metrics = tt.evaluate(X_opt, y_opt)
        tt.evaluate_equity(
            X_opt,
            y_opt,
            path="results/"
            + "{}_equity_{}_uncondtol={}_numfeatures={}.csv".format(district, "optimized", uncondtol, numfeatures),
        )
        tt.save_opt_policy(
            "policies/{}_optimized_uncondtol={}".format(district, uncondtol)
        )
        return metrics

    metrics = run(fold1, fold2)
    final_metrics = get_final_metrics(metrics)
    final_metrics["n_opt"] = len(fold2[0])
    final_metrics["policy"] = "optimized"
    final_metrics["numfeatures"] = numfeatures

    return final_metrics


def oracle_policy(district, uncondtol, pool=None, numfeatures=None):
    X, y, r, features = get_district_dataset([district], covariates=None)

    tt = OraclePovertyRateTargetedTransfers(c_bar=CBAR, unconditional_tolerance=uncondtol)
    tt.run_opt(y, r)
    metrics = tt.evaluate(X, y)
    tt.evaluate_equity(
            X,
            y,
            path="results/"
            + "{}_equity_{}_uncondtol={}.csv".format(district, "oracle", uncondtol),
        )

    final_metrics = get_final_metrics(metrics)
    final_metrics["n_opt"] = len(y)
    final_metrics["policy"] = "oracle"
    final_metrics["numfeatures"] = None

    return final_metrics


def conditional_optimized_policy(district, uncondtol, pool, numfeatures=None):

    # pooled = [d for d in POOLED_DISTRICTS if district != d]
    # TODO ADD COVARIATES
    # pooled=POOLED_DISTRICTS
    fold1, fold2, features = get_datasets(
        district, pool, covariates=get_covariates(district, numfeatures)
    )

    def run(fold_fit, fold_opt):
        tt = ConditionalTargetedTransfers(
            c_bar=CBAR, conditional_tolerance=uncondtol, method="qr"
        )
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, log_transform=True, low_dim=False, n_epochs=500)
        tt.run_opt(X_opt)
        metrics = tt.evaluate(X_opt, y_opt)
        tt.evaluate_equity(
            X_opt,
            y_opt,
            path="results/"
            + "{}_equity_{}_uncondtol={}_numfeatures={}.csv".format(district, "conditional_optimized", uncondtol, numfeatures),
        )
        return metrics

    metrics1 = run(fold1, fold2)
    final_metrics = get_final_metrics(metrics1)
    final_metrics["n_opt"] = len(fold2[0])
    final_metrics["policy"] = "conditional_optimized"
    final_metrics["numfeatures"] = numfeatures
    return final_metrics


def binary_conditional_optimized_policy(district, uncondtol, pool, numfeatures=None):

    # pooled = [d for d in POOLED_DISTRICTS if district != d]
    # TODO ADD COVARIATES
    # pooled=POOLED_DISTRICTS
    fold1, fold2, features = get_datasets(
        district, pool, covariates=get_covariates(district, numfeatures)
    )

    def run(fold_fit, fold_opt):
        tt = BinaryConditionalTargetedTransfers(
            c_bar=CBAR, conditional_tolerance=uncondtol, method="qr"
        )
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, log_transform=True, low_dim=False, n_epochs=500)
        tt.run_opt(X_opt)
        metrics = tt.evaluate(X_opt, y_opt)
        tt.evaluate_equity(
            X_opt,
            y_opt,
            path="results/"
            + "{}_equity_{}_uncondtol={}_numfeatures={}.csv".format(district, "binary_conditional_optimized", uncondtol, numfeatures),
        )
        return metrics

    metrics1 = run(fold1, fold2)
    final_metrics = get_final_metrics(metrics1)
    final_metrics["n_opt"] = len(fold2[0])
    final_metrics["policy"] = "binary_conditional_optimized"
    final_metrics["numfeatures"] = numfeatures
    return final_metrics



def get_final_metrics(metrics):
    keynames = [
        "initial_poverty_rate",
        "initial_poverty_gap",
        "post_transfer_poverty_gap",
        "post_transfer_poverty_rate",
        "policy_cost",
        "d",
        "unconditional_tolerance",
        "conditional_tolerance",
    ]
    final_metrics = {}
    for key in keynames:
        final_metrics[key] = metrics[key]
    return final_metrics
