import numpy as np
from scipy.stats import lognorm
from scipy.interpolate import interp1d
import itertools
from scipy.spatial import ConvexHull
from itertools import product


def compare_ratio(curr_p, old_p, new_p):
    slope1 = (new_p[1] - curr_p[1]) / (new_p[0] - curr_p[0])
    slope2 = (old_p[1] - curr_p[1]) / (old_p[0] - curr_p[0])
    return slope1 < slope2


class ConditionalDistribution:
    def __init__(self, x, scale, shape):
        self.x = x
        self.scale = scale
        self.shape = shape
        self.set_mode()
        self.set_inverses()

    def pdf(self, z):
        return lognorm.pdf(z, loc=self.x, scale=self.scale, s=self.shape)

    def cdf(self, z):
        return lognorm.cdf(z, loc=self.x, scale=self.scale, s=self.shape)

    def ppf(self, a):
        return lognorm.ppf(a, loc=self.x, scale=self.scale, s=self.shape)

    def set_mode(self):
        zs = np.linspace(0.0, self.x + 10, 10000)
        fs = self.pdf(zs)
        mode = zs[np.argmax(fs)]
        self.mode = mode

    def set_inverses(self):
        z1s = np.linspace(0.0, self.mode, 10000)
        z2s = np.linspace(self.mode, 30 + self.mode, 10000)
        p1s = self.pdf(z1s)
        p2s = self.pdf(z2s)

        self.inv1 = interp1d(p1s, z1s)
        self.inv2 = interp1d(p2s, z2s)

    def get_z(self, alpha, c_bar):
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
        z = self.get_z(alpha, c_bar)
        p = self.cdf(c_bar - z)
        return p

    def get_convex_hull(self, alpha, c_bar):
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
