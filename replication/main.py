from opt_targeted_transfers import (
    ConditionalTargetedTransfers,
    UnconditionalTargetedTransfers,
)
from data_loaders import get_dataset
from data_utils import split_data
from reporting import write_result
import argh


@argh.arg("--tolerance", nargs="+", type=float)
@argh.arg("--d", default=2)
@argh.arg("--country", default="malawi")
@argh.arg("--constraint", default="unconditional")
@argh.arg(
    "--method", default="qr"
)  # refers to quantile method if constraint = conditional
@argh.arg("--save", default="results")
def main(
    country="malawi",
    d=2,
    constraint="unconditional",
    method="qr",
    tolerance=None,
    save="malawi_results.csv",
):
    X, y, r, features = get_dataset(country)

    (X_train, y_train, r_train), (X_test, y_test, r_test) = split_data(
        X=X[:, :d], y=y, r=r, p=0.6
    )

    if constraint == "unconditional":
        tt = UnconditionalTargetedTransfers(name=country, c_bar=2.15)

    elif constraint == "conditional":
        tt = ConditionalTargetedTransfers(name=country, method=method, c_bar=2.15)

    fit_first = True
    if constraint == "conditional" and method == "qr":
        fit_first = False

    if fit_first:
        tt.fit(X_train, y_train, r_train)

        for tol in tolerance:
            tt.set_tolerance(tol)
            if constraint == "unconditional":
                tt.run_opt(
                    X_test,
                    r_test,
                    path=save + "{}_d={}_tol={}_opt.csv".format(country, d, tol),
                )
            else:
                tt.run_opt(X_test, r_test)
            res = tt.evaluate(X_test, y_test, r_test)
            write_result(save + "{}.csv".format(country), res)

    else:
        for tol in tolerance:
            tt.set_tolerance(tol)
            tt.fit(X_train, y_train, r_train)
            tt.run_opt(X_test, r_test)
            res = tt.evaluate(X_test, y_test, r_test)
            write_result(save + "{}.csv".format(country), res)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
