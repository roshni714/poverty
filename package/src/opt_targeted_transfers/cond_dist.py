import numpy as np
from scipy.stats import lognorm
import torch
from scipy.interpolate import interp1d


def get_lower_cvx_hull(tups):
    """
    Compute the lower convex hull of a set of points.

    :param tups: A list of 2D points represented as tuples.
    :type tups: list of tuples
    :return lower: An array representing the points on the lower convex hull,
             sorted by x-coordinate.
    :rtype: numpy.ndarray
    """

    def compare_ratio(curr_p, old_p, new_p):
        # Compares the change in cost-weight tradeoff from curr_p to old_p and from curr_p to new_p.
        # Returns True if new_p gives lower has lower slope (better tradeoff).

        slope1 = (new_p[1] - curr_p[1]) / (new_p[0] - curr_p[0])
        slope2 = (old_p[1] - curr_p[1]) / (old_p[0] - curr_p[0])
        return slope1 < slope2

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
    lower = np.array(lower)
    return lower


class ConditionalDistribution:
    """
    Represents a conditional distribution.

    :ivar inverses: Placeholder for inverse functions
    :ivar domains: Placeholder for domains of inverse functions.
    :vartype inverses: Any

    :return: A new ConditionalDistribution instance.
    :rtype: ConditionalDistribution
    """

    def __init__(self):
        """
        Initialize the ConditionalDistribution.

        :return: None
        """
        self.inverses = None
        self.domains = None

    def pdf(self, z):
        """
        Probability density function (pdf) of the distribution.

        :param z: The input value.
        :type z: Any
        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("pdf function not implemented")

    def cdf(self, z):
        """
        Cumulative distribution function (cdf) of the distribution.

        :param z: The input value.
        :type z: Any
        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("cdf function not implemented")

    def ppf(self, a):
        """
        Percent point function (inverse of cdf) of the distribution.

        :param a: The probability value.
        :type a: float
        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("ppf function not implemented")

    def expect(self, f):
        """
        Expected value of a given function under the distribution.

        :param f: The function for which the expected value is computed.
        :type f: callable
        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("expect function not implemented")

    def set_inverses(self):
        """
        Placeholder method to set inverse functions.

        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("set_inverses function not implemented")

    def get_nonboundary_alpha_valid_transfers(self, alpha, c_bar):
        if self.inverses is None:
            self.set_inverses()
        z = [c_bar]
        for i, domain in enumerate(self.domains):
            if alpha <= domain[1] and alpha >= domain[0]:
                inv = self.inverses[i]
                v = c_bar - inv(alpha)
                if v > 0 and v < c_bar:
                    z.append(v)
        z = np.array(sorted(z))
        return z

    def get_convex_hull(self, z, c_bar):
        p = self.cdf(c_bar - z)
        tups = list(zip(p, z))
        cvx_hull = get_lower_cvx_hull(tups)
        return cvx_hull


class LogNormalConditionalDistribution(ConditionalDistribution):
    """
    Represents a conditional distribution with log-normal distribution.

    :ivar loc: Location parameter of the log-normal distribution.
    :vartype loc: float
    :ivar scale: Scale parameter of the log-normal distribution.
    :vartype scale: float
    :ivar shape: Shape parameter of the log-normal distribution.
    :vartype shape: float
    :ivar mode: Mode of the distribution.
    :vartype mode: float

    :param loc: Location parameter of the log-normal distribution.
    :type loc: float
    :param scale: Scale parameter of the log-normal distribution.
    :type scale: float
    :param shape: Shape parameter of the log-normal distribution.
    :type shape: float
    """

    def __init__(self, loc, scale, shape):
        """
        Initialize the LogNormalConditionalDistribution.

        :param loc: Location parameter of the log-normal distribution.
        :type loc: float
        :param scale: Scale parameter of the log-normal distribution.
        :type scale: float
        :param shape: Shape parameter of the log-normal distribution.
        :type shape: float
        """
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.shape = shape

        mu = np.log(self.scale)
        sigma = self.shape
        self.mode = np.exp(mu - sigma**2) + self.loc

    def pdf(self, z):
        """
        Probability density function (pdf) of the log-normal distribution.

        :param z: The input value.
        :type z: float or numpy.ndarray
        :return: The probability density at z.
        :rtype: float or numpy.ndarray
        """
        return lognorm.pdf(z, loc=self.loc, scale=self.scale, s=self.shape)

    def cdf(self, z):
        """
        Cumulative distribution function (CDF) of the log-normal distribution.

        :param z: The input value.
        :type z: float or numpy.ndarray
        :return: The cumulative probability up to z.
        :rtype: float or numpy.ndarray
        """
        return lognorm.cdf(z, loc=self.loc, scale=self.scale, s=self.shape)

    def ppf(self, a):
        """
        Percent point function (inverse CDF) of the log-normal distribution.

        :param a: The probability value.
        :type a: float or numpy.ndarray
        :return: The value such that the CDF is equal to a.
        :rtype: float or numpy.ndarray
        """
        return lognorm.ppf(a, loc=self.loc, scale=self.scale, s=self.shape)

    def expect(self, f):
        """
        Expected value of a given function under the log-normal distribution.

        :param f: The function for which the expected value is computed.
        :type f: callable
        :return: The expected value of f under the distribution.
        :rtype: float
        """
        return lognorm.expect(f, loc=self.loc, scale=self.scale, args=(self.shape,))

    def set_inverses(self):
        """
        Computes the left (inv1) and right inverses (inv2) of the pdf.
        """
        z1s = np.linspace(self.loc, self.mode, 10000)
        z2s = np.linspace(self.mode, 30 * self.scale + self.mode, 10000)
        p1s = self.pdf(z1s)
        p2s = self.pdf(z2s)

        inv1 = interp1d(p1s, z1s, fill_value=(z1s[0], z1s[-1]), bounds_error=False)
        inv2 = interp1d(p2s, z2s, fill_value=(z2s[0], z2s[-1]), bounds_error=False)
        self.inverses = [inv1, inv2]
        self.domains = [np.array([min(p1s), max(p1s)]), np.array([min(p2s), max(p2s)])]


class NonparametricConditionalDistribution(ConditionalDistribution):
    """
    Represents a nonparametric conditional distribution.
    """

    def __init__(
        self, pdf_function, cdf_function, ppf_function, extrema, mode, outcome_range
    ):
        """
        Initialize a ConditionalDistribution object.

        :param pdf_function: The probability density function.
        :type pdf_function: callable
        :param cdf_function: The cumulative distribution function.
        :type cdf_function: callable
        :param ppf_function: The percent-point function (inverse of the CDF).
        :type ppf_function: callable
        :param extrema: The minimum and maximum possible values of the distribution.
        :type extrema: tuple
        :param mode: The mode of the distribution.
        :type mode: float
        :param outcome_range: The range of possible outcomes.
        :type outcome_range: tuple
        """
        super().__init__()
        self.cdf_function = cdf_function
        self.pdf_function = pdf_function
        self.ppf_function = ppf_function
        self.extrema = extrema
        self.mode = mode
        self.outcome_range = outcome_range

    def pdf(self, z):
        """
        Probability density function (pdf) of conditional distribution.

        :param z: The input value.
        :type z: float or numpy.ndarray
        :return: The probability density at z.
        :rtype: float or numpy.ndarray
        """
        return self.pdf_function(z)

    def cdf(self, z):
        """
        Cumulative distribution function (CDF) of the log-normal distribution.
        :param z: The input value.
        :type z: float or numpy.ndarray
        :return: The cumulative probability up to z.
        :rtype: float or numpy.ndarray
        """
        return np.clip(self.cdf_function(z), a_min=0.0, a_max=1.0)

    def ppf(self, a):
        """
        Percent point function (inverse CDF) of the log-normal distribution.

        :param a: The probability value.
        :type a: float or numpy.ndarray
        :return: The value such that the CDF is equal to a.
        :rtype: float or numpy.ndarray
        """
        return self.ppf_function(a)

    def set_inverses(self):
        """
        Computes the inverses of the pdf.
        """
        inverses = []
        domains = []
        for i in range(len(self.extrema) + 1):
            if i == 0:
                zs = np.linspace(self.outcome_range[0], self.extrema[i], 1000)
            elif i == len(self.extrema):
                zs = np.linspace(self.extrema[-1], self.outcome_range[1], 1000)
            else:
                zs = np.linspace(self.extrema[i - 1], self.extrema[i], 1000)

            ps = self.pdf(zs)
            inv = interp1d(ps, zs, fill_value=0.0, bounds_error=False)
            inverses.append(inv)
            domains.append(np.array([min(ps), max(ps)]))
        self.inverses = inverses
        self.domains = domains
