import numpy as np
from scipy.stats import lognorm
from scipy.interpolate import interp1d
import itertools
from scipy.spatial import ConvexHull
from itertools import product


def compare_ratio(curr_p, old_p, new_p):
    """
    Compares the change in objective-budget tradeoff from curr_p to old_p and from curr_p to new_p.
    Returns True if new_p gives lower has lower slope (better tradeoff).
    """

    slope1 = (new_p[1] - curr_p[1]) / (new_p[0] - curr_p[0])
    slope2 = (old_p[1] - curr_p[1]) / (old_p[0] - curr_p[0])
    return slope1 < slope2


class ConditionalDistribution:
    def __init__(self, loc, scale, shape):
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.mode = np.exp(np.log(self.scale) - (self.shape) ** 2) + self.loc
        self.inv1 = None
        self.inv2 = None

    #         zs = np.linspace(0.0, self.loc + 10, 10000)
    #         fs = self.pdf(zs)
    #         mode = zs[np.argmax(fs)]
    #         print(mode, self.mode)
    #        self.mode = mode

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

        self.inv1 = interp1d(p1s, z1s)
        self.inv2 = interp1d(p2s, z2s)

    def get_z(self, alpha, c_bar):
        """
        Computes the set of alpha-valid transfers.
        Returns a numpy array of alpha-valid transfers in sorted order.
        """
        if self.inv1 is None:
            self.set_inverses()
        z = [0, c_bar]
        if alpha < self.pdf(self.mode):
            v2 = c_bar - self.inv2(alpha)
            v1 = c_bar - self.inv1(alpha)
            if v1 > 0 and v1 < c_bar:
                z.append(v1)
            if v2 > 0 and v2 < c_bar:
                z.append(v2)
        z = np.array(sorted(z))
        return z

    def get_p(self, alpha, c_bar):
        """
        Computes the probability of that the post-transfer outcome is below  the poverty line for each of the alpha-valid transfer.
        """
        z = self.get_z(alpha, c_bar)
        p = self.cdf(c_bar - z)
        return p

    def get_convex_hull(self, alpha, c_bar):
        """
        Computes the lower convex hull of the alpha-valid transfers.
        """
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
