from opt_targeted_transfers import (
    ConditionalTargetedTransfers,
    UnconditionalTargetedTransfers,
    BinaryTargetedTransfers,
    OracleTargetedTransfers,
)
from data_loaders import get_datasets, get_district_dataset
from data_utils import aggregate_metrics

CBAR = 2.15

COVARIATE_LIST = [
    # "hh_f25",  # How much did you last pay for electricity?
    # "hh_f03a",  # Estimate the rent you could receive if you rented this property?
    "hhsize",  # Household size
    "hh_f12",  # What is your main source of cooking fuel?
    # "hh_f32",  # Total cost for <u>MTL</u> telephone service in the HH over the last period?\
    # "hh_f35_3", #Of the total cost of cellphone service for the household, how much was spent on airtime for all household members?
    # "hh_f04a", #How much do you pay to rent this property?<br><br>MK
    #    "hh_f40", #What is your main source of <u>drinking water in the <u>other season?
    "area",  # Rural/Urban division by region
    "hh_f06",  # what construction materials are used for dwelling
    # "hh_f37", #What was the total cost of drinking water for your household last month?
    #    "hh_h03b", #How many meals, including b/fast are taken per day in HH(Children 5-17yrs)
    #    "hh_h03a", #How many meals, including b/fast are taken per day in household? (Adults)
    # "hh_f26_2", #How satisfied are you with ESCOM?
    # "hh_g09", #Over the past one week (7 days), did any people that you did nonlist as household members eat any meals in your household?
    "af_bio_12_x",  # Annual Precipitation (mm)
    #    "hh_f41_2", #The last time your toilet was emptied
    #    "popdensity", #Pop density
    #    "hh_m00", #Did your household own or rent any farm implements, machinery and/or structures, such as hand hoe, panga knife, treadle pump, ox cart, tractor, plough, generator, chicken house, storage house, barn, etc... in the last 12 months?
]

RURAL_COVARIATE_LIST = ["hhsize", "hh_f06", "hh_f43", "hh_f41"]


def get_covariates(district):
    if district == "all":
        return COVARIATE_LIST
    else:
        return RURAL_COVARIATE_LIST


def saturation_policy(district, uncondtol, pool):
    fold1, fold2, features = get_datasets(district, pool, covariates=None)

    def run(fold_fit, fold_opt):
        tt = ConditionalTargetedTransfers(
            method="density", c_bar=CBAR, conditional_tolerance=uncondtol
        )
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, r_fit)
        tt.run_opt(X_opt, r_opt)
        metrics = tt.evaluate(X_opt, y_opt, r_opt)
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
    return final_metrics


def geographic_policy(district, uncondtol, pool=None):
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
        tt.fit(X_fit, y_fit, r_fit)
        tt.run_opt(
            X_opt,
            r_opt,
            path="results/{}_{}_uncondtol={}_opt.csv".format(
                district, "geographic", uncondtol
            ),
        )
        metrics = tt.evaluate(X_opt, y_opt, r_opt)
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
    return final_metrics


def binary_targeting_policy(district, uncondtol, pool):

    # "hh_t10" - what does head of house sleep on (bed/mattress)
    # "hh_f12" - what is your main source of cooking fuel
    # "hh_f41" - what kind of toilet
    # "hh_t03" - sufficient amount of clothing
    # "hh_t19" - did you not eat bc no food
    # "hh_t04" - concerning the standard of health care you received

    fold1, fold2, features = get_datasets(
        district, pool, covariates=get_covariates(district)
    )

    def run(fold_fit, fold_opt):
        tt = BinaryTargetedTransfers(c_bar=CBAR, unconditional_tolerance=uncondtol)
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, r_fit)
        tt.run_opt(X_opt, r_opt)
        metrics = tt.evaluate(X_opt, y_opt, r_opt)
        tt.evaluate_equity(
            X_opt,
            y_opt,
            path="results/"
            + "{}_equity_{}_uncondtol={}.csv".format(district, "binary", uncondtol),
        )
        tt.save_opt_policy(
            "policies/{}_binary_uncondtol={}".format(district, uncondtol)
        )
        return metrics

    metrics1 = run(fold1, fold2)
    final_metrics = get_final_metrics(metrics1)
    final_metrics["n_opt"] = len(fold2[0])
    final_metrics["policy"] = "binary"
    return final_metrics


def optimized_policy(district, uncondtol, pool):

    fold1, fold2, features = get_datasets(
        district, pool, covariates=get_covariates(district)
    )

    def run(fold_fit, fold_opt):
        tt = UnconditionalTargetedTransfers(
            c_bar=CBAR, unconditional_tolerance=uncondtol
        )
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, r_fit)
        tt.run_opt(
            X_opt,
            r_opt,
            path="results/{}_{}_uncondtol={}_opt.csv".format(
                district, "optimized", uncondtol
            ),
        )
        metrics = tt.evaluate(X_opt, y_opt, r_opt)
        tt.evaluate_equity(
            X_opt,
            y_opt,
            path="results/"
            + "{}_equity_{}_uncondtol={}.csv".format(district, "optimized", uncondtol),
        )
        tt.save_opt_policy(
            "policies/{}_optimized_uncondtol={}".format(district, uncondtol)
        )
        return metrics

    metrics = run(fold1, fold2)
    final_metrics = get_final_metrics(metrics)
    final_metrics["n_opt"] = len(fold2[0])
    final_metrics["policy"] = "optimized"
    return final_metrics


def oracle_policy(district, uncondtol, pool=None):
    X, y, r, features = get_district_dataset([district], covariates=None)

    tt = OracleTargetedTransfers(c_bar=CBAR, unconditional_tolerance=uncondtol)
    tt.run_opt(y, r)
    metrics = tt.evaluate(X, y, r)

    final_metrics = get_final_metrics(metrics)
    final_metrics["n_opt"] = len(y)
    final_metrics["policy"] = "oracle"
    return final_metrics


def conditional_optimized_policy(district, uncondtol, pool):

    # pooled = [d for d in POOLED_DISTRICTS if district != d]
    # TODO ADD COVARIATES
    # pooled=POOLED_DISTRICTS
    fold1, fold2, features = get_datasets(
        district, pool, covariates=get_covariates(district)
    )

    def run(fold_fit, fold_opt):
        tt = ConditionalTargetedTransfers(
            c_bar=CBAR, conditional_tolerance=uncondtol, method="qr"
        )
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, r_fit, low_dim=True, n_epochs=100)
        tt.run_opt(X_opt, r_opt)
        metrics = tt.evaluate(X_opt, y_opt, r_opt)
        return metrics

    metrics1 = run(fold1, fold2)
    final_metrics = get_final_metrics(metrics1)
    final_metrics["n_opt"] = len(fold2[0])
    final_metrics["policy"] = "conditional_optimized"
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
