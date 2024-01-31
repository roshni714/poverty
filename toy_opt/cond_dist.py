import numpy as np
from scipy.stats import lognorm
import itertools
from itertools import product
import torch
from scipy.interpolate import interp1d


def compare_ratio(curr_p, old_p, new_p):
    """
    Compares the change in objective-budget tradeoff from curr_p to old_p and from curr_p to new_p.
    Returns True if new_p gives lower has lower slope (better tradeoff).
    """

    slope1 = (new_p[1] - curr_p[1]) / (new_p[0] - curr_p[0])
    slope2 = (old_p[1] - curr_p[1]) / (old_p[0] - curr_p[0])
    return slope1 < slope2


class ConditionalDistribution:
    def __init__(self):
        self.inverses = None

    def pdf(self, z):
        raise NotImplementedError("pdf function not implemented")

    def cdf(self, z):
        raise NotImplementedError("cdf function not implemented")

    def ppf(self, a):
        raise NotImplementedError("ppf function not implemented")

    def expect(self, f):
        raise NotImplementedError("expect function not implemented")

    def set_inverses(self):
        raise NotImplementedError("set_inverses function not implemented")

    def get_z(self, alpha, c_bar):
        """
        Computes the set of alpha-valid transfers.
        Returns a numpy array of alpha-valid transfers in sorted order.
        """
        assert alpha > 0

        if self.inverses is None:
            self.set_inverses()
        z = [0, c_bar]
        for i, domain in enumerate(self.domains):
            if alpha <= domain[1] and alpha >= domain[0]:
                inv = self.inverses[i]
                v = c_bar - inv(alpha)
                if v > 0 and v < c_bar:
                    z.append(v)
        z = np.array(sorted(z))
        return z

    def get_p(self, alpha, c_bar):
        """
        Computes the probability of that the post-transfer outcome is below  the poverty line for each of the alpha-valid transfer.
        """
        assert alpha > 0
        z = self.get_z(alpha, c_bar)
        p = self.cdf(c_bar - z)
        return p

    def get_convex_hull(self, alpha, c_bar):
        """
        Computes the lower convex hull of the alpha-valid transfers.
        """
        assert alpha > 0
        z = self.get_z(alpha, c_bar)
        p = self.get_p(alpha, c_bar)
        tups = list(zip(p, z))
        sorted_tups = list(sorted(tups, key=lambda x: (x[0], x[1])))

        tups = []
        for j in range(len(sorted_tups)):
            if len(tups) == 0:
                tups.append(sorted_tups[0])
            else:
                if tups[-1][0] == sorted_tups[j][0]:
                    continue
                else:
                    tups.append(sorted_tups[j])

        if len(tups) == 1:
            return np.array(tups)

        lower = []
        for p in tups:
            while len(lower) >= 2 and compare_ratio(lower[-2], lower[-1], p):
                lower.pop()
            lower.append(p)
        return np.array(lower)


class LogNormalConditionalDistribution(ConditionalDistribution):
    def __init__(self, loc, scale, shape):
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.mode = np.exp(np.log(self.scale) - (self.shape) ** 2) + self.loc

    def pdf(self, z):
        return lognorm.pdf(z, loc=self.loc, scale=self.scale, s=self.shape)

    def cdf(self, z):
        return lognorm.cdf(z, loc=self.loc, scale=self.scale, s=self.shape)

    def ppf(self, a):
        return lognorm.ppf(a, loc=self.loc, scale=self.scale, s=self.shape)

    def set_inverses(self):
        """
        Computes the left (inv1) and right inverses of the pdf.
        """
        z1s = np.linspace(1e-10, self.mode, 10000)
        z2s = np.linspace(self.mode, 30 * self.scale + self.mode, 10000)
        p1s = self.pdf(z1s)
        p2s = self.pdf(z2s)

        inv1 = interp1d(p1s, z1s, fill_value=(z1s[0], z1s[-1]), bounds_error=False)
        inv2 = interp1d(p2s, z2s, fill_value=(z2s[0], z2s[-1]), bounds_error=False)
        self.inverses = [inv1, inv2]
        self.domains = [np.array([min(p1s), max(p1s)]), np.array([min(p2s), max(p2s)])]


class GLMSplineConditionalDistribution(ConditionalDistribution):
    def __init__(
        self, pdf_function, cdf_function, ppf_function, extrema, outcome_range
    ):
        super().__init__()
        self.cdf_function = cdf_function
        self.pdf_function = pdf_function
        self.ppf_function = ppf_function
        self.extrema = extrema
        self.mode = extrema[np.argmax([pdf_function(pt) for pt in extrema])].item()
        self.outcome_range = outcome_range

    def pdf(self, z):
        return self.pdf_function(z)

    def cdf(self, z):
        return np.clip(self.cdf_function(z), a_min=0.0, a_max=1.0)

    def ppf(self, a):
        return self.ppf_function(a)

    def set_inverses(self):
        """
        Computes the left (inv1) and right inverses of the pdf.
        """
        inverses = []
        domains = []
        for i, pt in enumerate(self.extrema):
            if i == 0:
                zs = np.linspace(self.outcome_range[0], pt, 1000)
            elif i == len(self.extrema) - 1:
                zs = np.linspace(pt, self.outcome_range[1], 1000)
            else:
                zs = np.linspace(self.extrema[i - 1], self.extrema[i], 1000)
            ps = self.pdf(zs)
            inv = interp1d(ps, zs, fill_value=0.0, bounds_error=False)
            inverses.append(inv)
            domains.append(np.array([min(ps), max(ps)]))
        self.inverses = inverses
        self.domains = domains


"""
        def right_pdf(z):
            if isinstance(z, np.ndarray):
                t = z < self.mode
                val = self.pdf(z)
                val[t] = 0.
                return val
            elif isinstance(z, float):
                if z >= self.mode:
                    return self.pdf(z)
                else:
                    return 0.

        def left_pdf(z):
            if isinstance(z, np.ndarray):
                t = z >= self.mode
                val = self.pdf(z)
                val[t] = 0.
                return val
            elif isinstance(z, float):
                if z < self.mode:
                    return self.pdf(z)
                else:
                    return 0.
       
        if alpha < self.pdf(self.mode):
            root1 = fsolve(lambda x: left_pdf(x) - alpha, self.mode- 10)
            root2 = fsolve(lambda x: right_pdf(x) - alpha, self.mode + 10)
            z.append(root1.item())
            z.append(root2.item())
"""
