from opt_targeted_transfers import (
    ConditionalTargetedTransfers,
    UnconditionalTargetedTransfers,
    HybridTargetedTransfers
)
from data_loaders import get_dataset
from data_utils import split_data
from reporting import write_result
import argh


@argh.arg("--uncondtol", nargs="+", type=float)
@argh.arg("--condtol", nargs="+", type=float)
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
    condtol=None,
    uncondtol=None,
    save="malawi_results.csv",
):
    X, y, r, features = get_dataset(country)

    (X_train, y_train, r_train), (X_test, y_test, r_test) = split_data(
        X=X[:, :d], y=y, r=None, p=0.6
    )  # for now not using sampling weights r

    if constraint == "unconditional":
        tt = UnconditionalTargetedTransfers(c_bar=2.15)
        
        tt.fit(X_train, y_train, r_train)
        for tol in uncondtol:
            tt.set_unconditional_tolerance(tol)
            tt.run_opt(
                    X_test,
                    r_test,
                    path=save + "{}_d={}_uncondtol={}_opt.csv".format(country, d, tol),
                )
            res = tt.evaluate(X_test, y_test, r_test)
            write_result(save + "{}.csv".format(country), res)
            tt.evaluate_equity(
                X_test,
                y_test,
                path=save
                + "equity_{}_{}_d={}_uncondtol={}.csv".format(country, tt.name, d, tol),
            )

    elif constraint == "conditional":
        tt = ConditionalTargetedTransfers(method=method, c_bar=2.15)

        if method == "density":
            tt.fit(X_train, y_train, r_train)
            for tol in condtol:
                tt.set_conditional_tolerance(tol)
                tt.run_opt(
                        X_test,
                        r_test,
                    )
                res = tt.evaluate(X_test, y_test, r_test)
                write_result(save + "{}.csv".format(country), res)
                tt.evaluate_equity(
                    X_test,
                    y_test,
                    path=save
                    + "equity_{}_{}_d={}_uncondtol={}_condtol={}.csv".format(country, tt.name, d, tol, tol),
                )
        elif method == "qr":
            for tol in condtol:
                tt.set_conditional_tolerance(tol)
                tt.fit(X_train, y_train, r_train)
                tt.run_opt(
                        X_test,
                        r_test,
                    )
                res = tt.evaluate(X_test, y_test, r_test)
                write_result(save + "{}.csv".format(country), res)
                tt.evaluate_equity(
                    X_test,
                    y_test,
                    path=save
                    + "equity_{}_{}_d={}_uncondtol={}_condtol={}.csv".format(country, tt.name, d, tol, tol),
                )
    elif constraint == "hybrid":

        tt = HybridTargetedTransfers(c_bar=2.15)
        tt.fit(X_train, y_train, r_train)
        
        for tol1 in uncondtol:
            for tol2 in condtol:
                tt.set_conditional_tolerance(tol2)
                tt.set_unconditional_tolerance(tol1)
                tt.run_opt(X_test,r_test, path=save + "{}_d={}_uncondtol={}_condtol={}_opt.csv".format(country, d, tol1, tol2))
                res = tt.evaluate(X_test, y_test, r_test)
                write_result(save + "{}.csv".format(country), res)
                tt.evaluate_equity(
                    X_test,
                    y_test,
                    path=save
                    + "equity_{}_{}_d={}_uncondtol={}_condtol={}.csv".format(country, tt.name, d, tol1, tol2),
                )

        


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
