from queue import PriorityQueue
import numpy as np


def solve_fractional_knapsack_problem(xs, p_xs, convex_hulls, budget):
    """
    Priority queue algorithm of Svedrup et al 2023.
    """

    total_gain = 0
    total_spend = 0
    pq = PriorityQueue()
    assignments = {x_idx: [] for x_idx in range(len(xs))}
    for i in range(len(xs)):
        init_point = convex_hulls[i][0]
        total_spend += init_point[0] * p_xs[i]
        total_gain += init_point[1] * p_xs[i]

        if len(convex_hulls[i]) > 1:
            second_point = convex_hulls[i][1]
            ratio = (second_point[1] - init_point[1]) / (
                second_point[0] - init_point[0]
            )
            pq.put((ratio, (i, 1)))
        assignments[i] = [(init_point[1], 1)]

    while total_spend < budget and pq.qsize() > 0:
        ratio, (x_idx, hull_idx) = pq.get()
        # Remove previous assignment
        prev_spend = convex_hulls[x_idx][hull_idx - 1][0] * p_xs[x_idx]
        prev_gain = convex_hulls[x_idx][hull_idx - 1][1] * p_xs[x_idx]
        total_spend -= prev_spend
        total_gain -= prev_gain

        # Add new assignment
        curr_spend = convex_hulls[x_idx][hull_idx][0] * p_xs[x_idx]
        curr_gain = convex_hulls[x_idx][hull_idx][1] * p_xs[x_idx]

        total_gain += curr_gain
        total_spend += curr_spend
        assignments[x_idx] = [(convex_hulls[x_idx][hull_idx][1], 1)]

        if total_spend > budget:
            # Fractional allocation
            total_spend -= curr_spend
            total_gain -= curr_gain

            remainder = budget - total_spend

            prev_spend = convex_hulls[x_idx][hull_idx - 1][0]
            prev_gain = convex_hulls[x_idx][hull_idx - 1][1]

            curr_spend = convex_hulls[x_idx][hull_idx][0]
            curr_gain = convex_hulls[x_idx][hull_idx][1]

            c = np.minimum(
                (remainder / p_xs[x_idx] - curr_spend) / (prev_spend - curr_spend), 1.0
            )

            assignments[x_idx] = [
                (convex_hulls[x_idx][hull_idx - 1][1], c),
                (convex_hulls[x_idx][hull_idx][1], 1 - c),
            ]
            total_spend += (c * prev_spend + (1 - c) * curr_spend) * p_xs[x_idx]
            total_gain += (c * prev_gain + (1 - c) * curr_gain) * p_xs[x_idx]

            assert round(total_spend, 3) == budget

        if hull_idx != len(convex_hulls[x_idx]) - 1:
            next_hull_idx = hull_idx + 1
            next_point = convex_hulls[x_idx][next_hull_idx]
            curr_point = convex_hulls[x_idx][hull_idx]
            ratio = (next_point[1] - curr_point[1]) / (next_point[0] - curr_point[0])
            pq.put((ratio, (x_idx, next_hull_idx)))

    return assignments, total_gain
