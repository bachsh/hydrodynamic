### Set of quantities and functions related to the Hydrodynamic limit of the KLS model.

import numpy as np

def kls_lambda(rho, epsilon):
    e4b = (1+epsilon)/(1-epsilon)
    four_rho_1_rho = 4*rho*(1-rho)
    return (1+np.sqrt((2*rho-1)**2 + four_rho_1_rho/e4b)) / np.sqrt(four_rho_1_rho)

def kls_J(rho, epsilon, delta):
    lam = kls_lambda(rho, epsilon)
    return (lam*(1+delta*(1-2*rho)) - epsilon*np.sqrt(4*rho*(1-rho))) / lam**3

def kls_chi(rho, epsilon):
    return rho*(1-rho)*np.sqrt((2*rho-1)**2 + 4*rho*(1-rho)*(1-epsilon)/(1+epsilon))

def kls_D(rho, epsilon, delta):
    if epsilon==1:
        return (rho<=0.5)*(1+delta)/(1-rho)**2 + (rho>0.5)*(1-delta)/(rho)**2
    return kls_J(rho, epsilon, delta) / kls_chi(rho, epsilon)

def kls_sigma(rho, epsilon, delta):
    if epsilon==1:
        return (rho<=0.5)*(1+delta)*rho*(1-2*rho)/(1-rho) + (rho>0.5)*(1-delta)*(1-rho)*(2*rho-1)/rho,
    return 2*kls_J(rho, epsilon, delta)

