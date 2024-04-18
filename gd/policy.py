from opt_targeted_transfers import (
    ConditionalTargetedTransfers,
    UnconditionalTargetedTransfers,
    BinaryTargetedTransfers,
    OracleTargetedTransfers,
)
from data_loaders import get_dataset
from data_utils import split_data, aggregate_metrics

CBAR = 2.15


def saturation_policy(district, uncondtol):
    X, y, r, features = get_dataset(district, covariates=None)

    fold1, fold2 = split_data(X=X, y=y, r=r, p=0.5)

    def run(fold_fit, fold_opt):
        tt = ConditionalTargetedTransfers(
            method="density", c_bar=CBAR, conditional_tolerance=uncondtol
        )
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, r_fit)
        tt.run_opt(X_opt, r_opt)
        metrics = tt.evaluate(X_opt, y_opt, r_opt)
        return metrics

    metrics1 = run(fold1, fold2)
    metrics2 = run(fold2, fold1)
    final_metrics = aggregate_metrics(metrics1, metrics2)
    final_metrics["policy"] = "saturation"
    return final_metrics


def geographic_policy(district, uncondtol):

    X, y, r, features = get_dataset(district, covariates=["ea_id"])

    fold1, fold2 = split_data(X=X, y=y, r=r, p=0.5)

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
        return metrics

    metrics1 = run(fold1, fold2)
    metrics2 = run(fold2, fold1)
    final_metrics = aggregate_metrics(metrics1, metrics2)
    final_metrics["policy"] = "geographic"
    return final_metrics


def binary_targeting_policy(district, uncondtol):

    # "hh_t10" - what does head of house sleep on (bed/mattress)
    # "hh_f12" - what is your main source of cooking fuel
    # "hh_f41" - what kind of toilet
    # "hh_t03" - sufficient amount of clothing
    # "hh_t19" - did you not eat bc no food
    # "hh_t04" - concerning the standard of health care you received

    X, y, r, features = get_dataset(
        district,
        covariates=[
            "ea_id",
            "hh_t10",
            "hh_f12",
            "hh_f41",
            "hh_t03",
            "hh_t19",
            "hh_t04",
        ],
    )

    fold1, fold2 = split_data(X=X, y=y, r=r, p=0.5)

    def run(fold_fit, fold_opt):
        tt = BinaryTargetedTransfers(c_bar=CBAR, unconditional_tolerance=uncondtol)
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, r_fit)
        tt.run_opt(X_opt, r_opt)
        metrics = tt.evaluate(X_opt, y_opt, r_opt)
        return metrics

    metrics1 = run(fold1, fold2)
    metrics2 = run(fold2, fold1)
    final_metrics = aggregate_metrics(metrics1, metrics2)
    final_metrics["policy"] = "binary_targeted"
    return final_metrics


def optimized_policy(district, uncondtol):

    # TODO ADD COVARIATES
    X, y, r, features = get_dataset(
        district,
        covariates=[
            "ea_id",
            "hh_t10",
            "hh_f12",
            "hh_f41",
            "hh_t03",
            "hh_t19",
            "hh_t04",
        ],
    )

    fold1, fold2 = split_data(X=X, y=y, r=r, p=0.5)

    def run(fold_fit, fold_opt):
        tt = UnconditionalTargetedTransfers(
            c_bar=CBAR,
            unconditional_tolerance=uncondtol,
            path="results/{}_{}_uncondtol={}_opt.csv".format(
                district, "optimized", uncondtol
            ),
        )
        X_fit, y_fit, r_fit = fold_fit
        X_opt, y_opt, r_opt = fold_opt
        tt.fit(X_fit, y_fit, r_fit)
        tt.run_opt(X_opt, r_opt)
        metrics = tt.evaluate(X_opt, y_opt, r_opt)
        return metrics

    metrics1 = run(fold1, fold2)
    metrics2 = run(fold2, fold1)
    final_metrics = aggregate_metrics(metrics1, metrics2)
    final_metrics["policy"] = "optimized"
    return final_metrics


def oracle_policy(district, uncondtol):
    X, y, r, features = get_dataset(district, covariates=None)

    tt = OracleTargetedTransfers(c_bar=CBAR, unconditional_tolerance=uncondtol)
    tt.run_opt(y, r)
    metrics = tt.evaluate(X, y, r)
    del metrics["nclass"]
    del metrics["d"]
    del metrics["method"]
    metrics["policy"] = "oracle"
    return metrics
