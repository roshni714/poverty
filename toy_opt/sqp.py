import cvxpy as cp
import numpy as np
from scipy.stats import norm

np.random.seed(0)

d = 10
xs = np.linspace(-1., 1., d)
p = np.linspace(0.1, 1.1, d)
p /= p.sum()
c_bar = 0.3
eps = 0.05

nu = cp.Variable(d) 

lam = np.array([100.])

def get_f(xs, c_bar, g):
    return norm.pdf(c_bar - np.exp(g) - xs ** 2)

def get_f_prime(xs, c_bar, g):
    return - (c_bar - np.exp(g) - xs ** 2) * norm.pdf(c_bar - np.exp(g) - xs ** 2) 

def get_probability(xs, p, c_bar, g):
    return np.array([norm.cdf(c_bar - np.exp(g) - xs**2).T @ p])


#def find_starting_point():
#    Q = np.zeros(d)
 
#    while np.sum(Q > 0)  != d:
#        g = np.random.rand(d)
#        f = get_f(xs, c_bar, g)
#        f_prime = get_f_prime(xs, c_bar, g)
#        probab = get_probability(xs, c_bar, g)
#        t= np.exp(g)
#        Q = lam * (t ** 2) * f_prime + (t * (1 - lam * f))
#    return g


#g = find_starting_point() 
g = np.log(np.random.uniform(0.01, 5., d)) 
#g = np.log(c_bar - norm.ppf(eps, loc=xs ** 2)) #find_starting_point()

print("found starting point")
def create_problem(g, lam):
    f = get_f(xs, c_bar, g)
    f_prime = get_f_prime(xs, c_bar, g)
    probab = get_probability(xs, p, c_bar, g)
    t= np.exp(g)

    Q = lam * (t ** 2) * f_prime * p + (t * (1 - lam * f)) * p
    Q = np.diag(Q)

    obj_part1 = ((t * p).T @ nu)
    obj_part2 = (1/2) *  cp.quad_form(nu, Q)

    obj = obj_part1 + obj_part2
    constraints = [probab - eps - (t * f * p).T @ nu == 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    return prob, constraints

iterations = 100

def phi(z, mu=10000):
    cost = np.exp(z).T @ p 
    viol = mu * np.abs(get_probability(xs, p, c_bar, z) - eps)
    return cost + viol
   

def backtracking_linesearch(g, nu):
    alpha = 0.1
    phi_g = phi(g)
    while phi(g + alpha * nu) > phi_g:
        alpha *= 0.1
    return alpha 

merits = []
probabs = []
objs = []
for k in range(iterations):
    print("lam", lam)
    #Log
    merits.append(phi(g))
    probabs.append(get_probability(xs, p, c_bar, g))
    objs.append( np.mean(np.exp(g))) 
    print("iteration {}, merit {}, probabs {}, obj {}".format(k, phi(g), get_probability(xs, p, c_bar,  g), np.mean(np.exp(g))))

    #Solve
    prob, constraints = create_problem(g, lam)
    prob.solve()
    alpha = backtracking_linesearch(g, nu.value)
    lam_qp = constraints[0].dual_value
    print("stepsize", alpha)    
    #Update values
    lam = max(lam + alpha * (lam_qp - lam), 0.1)
    g += alpha * nu.value
    
#print(merits)
print(probabs)
print(objs)
print(np.exp(g).T @ p)
