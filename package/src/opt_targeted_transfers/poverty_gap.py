class OraclePovertyGapPolicy:
    def __init__(self, budget, c_bar=2.15):
        self.c_bar = c_bar
        self.budget = budget
        self.opt_policy = None
        self.name = "oracle_poverty_gap"
        self.nclass = None

    def run_opt(X_test, y_test, r_test=None):
        dataset = Dataset(X=None, y=y_test, r=r_test)

        oracle_policy = run_oracle_poverty_rate(
            dataset, c_bar=self.c_bar, tolerance=self.unconditional_tolerance
        )
        self.opt_policy = oracle_policy
        return oracle_policy

    def save_opt_policy(self, name):
        if self.opt_policy is None:
            assert False, "Need to run opt first"
        pickle.dump(
            self.opt_policy,
            open("{}.pickle".format(name), "wb"),
        )

    def evaluate(self, X_test, y_test, r_test=None):
        """
        Evaluate optimal policy.

        :param X_test: The input features of the test data.
        :type X_test: numpy.ndarray
        :param y_test: The target values of the test data.
        :type y_test: numpy.ndarray
        :param r_test: The response variable of the test data. Defaults to None.
        :type r_test: numpy.ndarray or None
        :return: A dictionary of evaluation results.
        :rtype: dict
        """

        if self.opt_policy is None:
            assert False, "Need to first run optimization"

        dataset = Dataset(X_test, y_test, r_test)
        if "oracle" in self.name:
            result = post_transfer_metrics(
                dataset, self.opt_policy, self.c_bar, oracle=True
            )
        else:
            result = post_transfer_metrics(dataset, self.opt_policy, self.c_bar)
        if len(X_test.shape) > 1:
            d = X_test.shape[1]
        else:
            d = 0
        result.update(
            {
                "method": self.name,
                "unconditional_tolerance": self.unconditional_tolerance,
                "conditional_tolerance": self.conditional_tolerance,
                "d": d,
                "nclass": self.nclass,
            }
        )
        return result
