import numpy as np
from tqdm import tqdm
import os
import bisect

from opt_targeted_transfers.priority_queue import PriorityQueue
from opt_targeted_transfers.reporting import write_result


def solve_fractional_mc_knapsack_problem(p_xs, convex_hulls, tolerance):
    """
    Priority queue algorithm of Svedrup et al 2023.

    :param p_xs: A numpy.array of weights.
    :type p_xs: list[(float, float)]
    :param convex_hulls: A list of convex hulls representing feasible solutions.
    :type convex_hulls: list[numpy.ndarray]
    :param tolerance: Tolerance for poverty rate.
    :type tolerance: float
    :return: A tuple including assignments, total_gain (total transfer amount),
             total_spend (tolerance), threshold cost-benefit ratio, threshold probability.
    :rtype: (dict, float, float, float, float)
    """

    total_gain = 0
    total_spend = 0
    pq = PriorityQueue()
    assignments = {x_idx: [] for x_idx in range(len(p_xs))}
    for i in range(len(p_xs)):
        init_point = convex_hulls[i][0]
        total_spend += init_point[0] * p_xs[i]
        total_gain += init_point[1] * p_xs[i]

        if len(convex_hulls[i]) > 1:
            second_point = convex_hulls[i][1]
            ratio = (second_point[1] - init_point[1]) / (
                second_point[0] - init_point[0]
            )
            pq.put(ratio, (i, 1))
        assignments[i] = [(init_point[1], 1)]

    etas = [-float("inf")]
    metadata = [None]
    lamb = 0.0
    while total_spend < tolerance and pq:
        ratio, tups = pq.get()
        # Remove previous assignment
        prev_spend = sum(
            [
                convex_hulls[x_idx][hull_idx - 1][0] * p_xs[x_idx]
                for x_idx, hull_idx in tups
            ]
        )
        prev_gain = sum(
            [
                convex_hulls[x_idx][hull_idx - 1][1] * p_xs[x_idx]
                for x_idx, hull_idx in tups
            ]
        )
        total_spend -= prev_spend
        total_gain -= prev_gain

        # Add new assignment
        curr_spend = sum(
            [convex_hulls[x_idx][hull_idx][0] * p_xs[x_idx] for x_idx, hull_idx in tups]
        )
        curr_gain = sum(
            [convex_hulls[x_idx][hull_idx][1] * p_xs[x_idx] for x_idx, hull_idx in tups]
        )

        total_gain += curr_gain
        total_spend += curr_spend
        for x_idx, hull_idx in tups:
            assignments[x_idx] = [(convex_hulls[x_idx][hull_idx][1], 1)]

        etas.append(ratio)
        metadata.append(tups)
        if total_spend > tolerance:
            # Fractional allocation
            total_spend -= curr_spend
            total_gain -= curr_gain

            remainder = tolerance - total_spend

            prev_spend = sum(
                [
                    convex_hulls[x_idx][hull_idx - 1][0] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )
            prev_gain = sum(
                [
                    convex_hulls[x_idx][hull_idx - 1][1] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )

            curr_spend = sum(
                [
                    convex_hulls[x_idx][hull_idx][0] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )
            curr_gain = sum(
                [
                    convex_hulls[x_idx][hull_idx][1] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )

            c = np.minimum((remainder - curr_spend) / (prev_spend - curr_spend), 1.0)
            lamb = c
            for x_idx, hull_idx in tups:
                assignments[x_idx] = [
                    (convex_hulls[x_idx][hull_idx - 1][1], c),
                    (convex_hulls[x_idx][hull_idx][1], 1 - c),
                ]
            total_spend += c * prev_spend + (1 - c) * curr_spend
            total_gain += c * prev_gain + (1 - c) * curr_gain
            return assignments, total_gain, total_spend, etas[-1], lamb

        for x_idx, hull_idx in tups:
            if hull_idx != len(convex_hulls[x_idx]) - 1:
                next_hull_idx = hull_idx + 1
                next_point = convex_hulls[x_idx][next_hull_idx]
                curr_point = convex_hulls[x_idx][hull_idx]
                ratio = (next_point[1] - curr_point[1]) / (
                    next_point[0] - curr_point[0]
                )
                pq.put(ratio, (x_idx, next_hull_idx))

    return assignments, total_gain, total_spend, etas[-1], lamb


def get_alpha_transfer_function(
    alpha, c_bar, eta, lamb, compute_cond_density, min_transfer_function=None
):
    """
    Compute the transfer function.

    :param alpha: The alpha value.
    :type alpha: float
    :param c_bar: The poverty line.
    :type c_bar: float
    :param eta: The threshold cost-benefit ratio.
    :type eta: float
    :param lamb: The threshold probability.
    :type lamb: float
    :param compute_cond_density: A function to compute the conditional density.
    :type compute_cond_density: Callable[[np.ndarray], np.ndarray]
    :return: The transfer function.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def t(X_test):
        cond_densities = compute_cond_density(X_test)
        assignments = {x_idx: [] for x_idx in range(len(X_test))}
        cvx_hulls = get_alpha_convex_hulls(
            alpha,
            c_bar,
            cond_dists=cond_densities,
            min_transfer_function=min_transfer_function,
        )
        min_transfer_values = np.zeros(len(cond_densities))
        if min_transfer_function:
            min_transfer_values = min_transfer_function(cond_densities)

        for j, cond_density in enumerate(cond_densities):
            cvx_hull = cvx_hulls[j]
            ratios = np.zeros(len(cvx_hull)).astype(np.float64)
            ratios[0] = -np.inf
            for i in range(len(cvx_hull) - 1):
                p1 = cvx_hull[i]
                p2 = cvx_hull[i + 1]
                ratios[i + 1] = (p2[1] - p1[1]) / (p2[0] - p1[0])
            idx = bisect.bisect_left(ratios, eta)

            if (
                idx > 0
                and idx < len(ratios)
                and ratios[idx - 1] < eta
                and ratios[idx] > eta
            ):
                assignments[j] = [(cvx_hull[idx - 1][1], 1.0)]
            elif idx < len(ratios) and ratios[idx] == eta:
                assignments[j] = [
                    (cvx_hull[idx - 1][1], lamb),
                    (cvx_hull[idx][1], 1 - lamb),
                ]
            else:
                assignments[j] = [(min_transfer_values[j], 1.0)]

        return assignments

    return t


def get_transfer_function(
    transfer_amts, c_bar, eta, lamb, compute_cond_density, deterministic=False
):
    """
    Compute the transfer function.

    :param c_bar: The poverty line.
    :type c_bar: float
    :param eta: The threshold cost-benefit ratio.
    :type eta: float
    :param lamb: The threshold probability.
    :type lamb: float
    :param compute_cond_density: A function to compute the conditional density.
    :type compute_cond_density: Callable[[np.ndarray], np.ndarray]
    :return: The transfer function.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def t(X_test):
        cond_densities = compute_cond_density(X_test)
        assignments = {x_idx: [] for x_idx in range(len(X_test))}

        for j, cond_density in enumerate(cond_densities):
            cvx_hull = cond_density.get_convex_hull(z=transfer_amts, c_bar=c_bar)
            ratios = np.zeros(len(cvx_hull)).astype(np.float64)
            ratios[0] = -np.inf
            for i in range(len(cvx_hull) - 1):
                p1 = cvx_hull[i]
                p2 = cvx_hull[i + 1]
                ratios[i + 1] = (p2[1] - p1[1]) / (p2[0] - p1[0])
            idx = bisect.bisect_left(ratios, eta)

            if (
                idx > 0
                and idx < len(ratios)
                and ratios[idx - 1] < eta
                and ratios[idx] > eta
            ):
                assignments[j] = [(cvx_hull[idx - 1][1], 1.0)]
            elif idx < len(ratios) and ratios[idx] == eta:
                if deterministic:
                    assignments[j] = [
                        (max(cvx_hull[idx - 1][1], cvx_hull[idx][1]), 1.0)
                    ]
                else:
                    assignments[j] = [
                        (cvx_hull[idx - 1][1], lamb),
                        (cvx_hull[idx][1], 1 - lamb),
                    ]
            else:
                assignments[j] = [(0.0, 1.0)]

        return assignments

    return t


def check_assignments_are_equal(assignment1, assignment2):
    assert assignment1.keys() == assignment2.keys()

    for key in assignment1.keys():
        val1 = assignment1[key]
        val2 = assignment2[key]
        assert val1 == val2, "error at key {} bc {} != {}".format(key, val1, val2)


def check_knapsack_feasibility(
    dataset,
    cond_densities,
    unconditional_tolerance,
    raw_min_transfer_function,
    c_bar,
    max_transfer_value,
):

    if raw_min_transfer_function is not None:
        raw_min_transfer_values = raw_min_transfer_function(cond_densities)
        if any(raw_min_transfer_values > max_transfer_value):
            return False
    probs = np.array(
        [
            cond_density.cdf(c_bar - max_transfer_value)
            for cond_density in cond_densities
        ]
    )
    prob_total = np.sum(probs * dataset.r).item()
    if prob_total > unconditional_tolerance:
        return False
    else:
        return True


def get_convex_hulls(c_bar, cond_dists, transfer_amts, min_transfer_function=None):
    transfer_values = []
    min_transfer_values = np.zeros(len(cond_dists))
    if min_transfer_function is not None:
        min_transfer_values = min_transfer_function(cond_dists)

    for i in range(len(cond_dists)):
        tv = list(transfer_amts[i])
        tv = [t for t in tv if t > min_transfer_values[i]]
        tv.append(min_transfer_values[i])
        transfer_values.append(np.array(sorted(tv)))

    cvx_hulls = [
        cond_dists[i].get_convex_hull(tv, c_bar) for i, tv in enumerate(transfer_values)
    ]
    return cvx_hulls


def compute_cost(train_dataset, policy):
    assignments = policy(train_dataset.X)

    total_cost = 0.0
    for i in range(len(train_dataset)):
        cost = 0.0
        for j in range(len(assignments[i])):
            cost += assignments[i][j][1] * assignments[i][j][0]
        total_cost += cost * train_dataset.r[i]

    return total_cost


def compute_opt_policy_knapsack(
    train_dataset,
    cond_dists,
    raw_min_transfer_function,
    tolerance,
    transfer_amts,
    c_bar,
    compute_cond_density,
    deterministic=False,
):

    feasible = check_knapsack_feasibility(
        train_dataset,
        cond_dists,
        tolerance,
        raw_min_transfer_function,
        c_bar,
        max(transfer_amts),
    )
    if not feasible:
        return False

    else:
        if raw_min_transfer_function is not None:

            def min_transfer_function(cond_densities):
                raw_min_transfer_values = raw_min_transfer_function(cond_densities)
                if len(transfer_amts) == 2 and transfer_amts[0] == 0.0:
                    min_transfer_values = (
                        np.array(raw_min_transfer_values) > 0
                    ).astype(float) * transfer_amts[1]
                else:
                    raise NotImplementedError
                return min_transfer_values

        else:
            min_transfer_function = None

        cvx_hulls = get_convex_hulls(
            c_bar,
            cond_dists,
            [transfer_amts for i in range(len(train_dataset))],
            min_transfer_function,
        )
        (opt_assignment, total_transfer, prob_below_line, eta, lamb) = (
            solve_fractional_mc_knapsack_problem(train_dataset.r, cvx_hulls, tolerance)
        )
        t = get_transfer_function(
            transfer_amts=transfer_amts,
            c_bar=c_bar,
            eta=eta,
            lamb=lamb,
            compute_cond_density=compute_cond_density,
            deterministic=deterministic,
        )

        new_total_transfer = compute_cost(train_dataset, t)
        return t, new_total_transfer


def get_alpha_convex_hulls(alpha, c_bar, cond_dists, min_transfer_function=None):
    nonboundary_transfer_values = [
        c_dist.get_nonboundary_alpha_valid_transfers(alpha, c_bar)
        for c_dist in cond_dists
    ]
    return get_convex_hulls(
        c_bar, cond_dists, nonboundary_transfer_values, min_transfer_function
    )


def compute_alpha_opt_policies(
    train_dataset,
    compute_cond_density,
    tolerance,
    c_bar,
    min_alpha=None,
    max_alpha=None,
    n_alpha=200,
    min_transfer_function=None,
    path="sim",
):
    """
    Compute alpha-optimal policies for a given training dataset.

    :param train_dataset: The training dataset.
    :type train_dataset: Dataset
    :param compute_cond_density: A function to compute the conditional density.
    :type compute_cond_density: Callable[[np.ndarray], np.ndarray]
    :param tolerance: Tolerance for poverty rate.
    :type tolerance: float
    :param c_bar: The poverty line.
    :type c_bar: float
    :param min_alpha: The minimum value of alpha for optimization. Defaults to None.
    :type min_alpha: float or None
    :param max_alpha: The maximum value of alpha for optimization. Defaults to None.
    :type max_alpha: float or None
    :param n_alpha: The number of alpha values to consider. Defaults to 200.
    :type n_alpha: int
    :param path: The path to save the simulation results. Defaults to "sim".
    :type path: str
    """
    cond_dists = compute_cond_density(train_dataset.X)

    total_transfers = []
    opt_policies = []
    if max_alpha is None:
        max_alpha = np.quantile([dist.pdf(dist.mode) for dist in cond_dists], 0.97)
    if min_alpha is None:
        min_alpha = max_alpha / 1000

    alphas = np.linspace(min_alpha, max_alpha, n_alpha)
    print("Alpha range: {}, {}".format(alphas[0], alphas[-1]))
    results_file = path

    if os.path.exists(results_file):
        os.remove(results_file)

    for alpha in tqdm(alphas):
        cvx_hulls = get_alpha_convex_hulls(
            alpha=alpha,
            c_bar=c_bar,
            cond_dists=cond_dists,
            min_transfer_function=min_transfer_function,
        )

        (
            opt_assignment,
            total_transfer,
            prob_below_line,
            eta,
            lamb,
        ) = solve_fractional_mc_knapsack_problem(train_dataset.r, cvx_hulls, tolerance)
        t_alpha = get_alpha_transfer_function(
            alpha,
            c_bar,
            eta,
            lamb,
            compute_cond_density=compute_cond_density,
            min_transfer_function=min_transfer_function,
        )

        # prox_assignment = t_alpha(train_dataset.X)
        # check_assignments_are_equal(opt_assignment, prox_assignment)

        result = {
            "alpha": alpha,
            "total_transfer": total_transfer,
            "prob_below_line": prob_below_line,
        }
        total_transfers.append(total_transfer)
        opt_policies.append(t_alpha)
        if results_file is not None:
            write_result(results_file, result)
    return opt_policies, total_transfers, alphas
