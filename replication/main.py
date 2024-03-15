from opt_targeted_transfers import ConditionalTargetedTransfers, UnconditionalTargetedTransfers
from data_loaders import get_dataset
from data_utils import split_data
import argh

@argh.arg("--tolerance", nargs="+", type=float)
@argh.arg("--d", default=2)
@argh.arg("--country", default="malawi")
@argh.arg("--constraint",  default="uncondtional")
@argh.arg("--quantile_method", default="qr") # refers to quantile method if constraint = conditional
@argh.arg("--save", default="malawi_results.csv")
def main(country="malawi", d=2, tolerance=None, constraint="unconditional", quantile_method = "qr"):
    X, y, r, features = get_dataset(country)

    (X_train, y_train, r_train), (X_test, y_test, r_test) = split_data(X=X[:, :d], y=y, r=r, p=0.6)

    fit_first = False

    if constraint == "unconditional":   
        tt = UnconditionalTargetedTransfer(name=country, c_bar=2.15)
        fit_first = True

    elif constraint == "conditional":
        tt = ConditionalTargetedTransfer(name=country, quantile_method=quantile_method, c_bar=2.15)

        if method == "density"
            fit_first = True

    if fit_first:
        tt.fit(X_train, y_train, r_train)

        for tol in tolerance:
            tt.set_tolerance(tol)
            tt.run_opt(X_test, r_test)
            res = tt.evaluate(X_test, y_test, r_test)
            write_results(res, save)


    else:
        for tol in tolerance:
            tt.set_tolerance(tol)
            tt.fit(X_train, y_train, r_train)
            tt.run_opt(X_test, r_test)
            res = tt.evaluate(X_test, y_test, r_test)
            write_results(res, save)


 

    
    
