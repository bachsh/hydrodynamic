import numpy as np
import scipy.optimize as opti


def steadyStateEquation(rhoBulk, rhoL, rhoR, D, sigma, E, dx):
    """A helper function for solving steady state profile equation"""
    # rho = [rhoL] + rhoBulk + [rhoR]
    rho = rhoBulk
    rho = np.insert(rho, 0, rhoL)
    rho = np.append(rho, rhoR)

    Drho = D(rho)
    dxDrho = np.zeros_like(rho)
    dxDrho[1:-1] = Drho[2:]-Drho[:-2]
    dxDrho = dxDrho/(2*dx)

    dxrho = np.zeros_like(rho)
    dxrho[1:-1] = rho[2:]-rho[:-2]
    dxrho = dxrho/(2*dx)

    dx2rho = np.zeros_like(rho)
    dx2rho[1:-1] = rho[2:]-2*rho[1:-1]+rho[:-2]
    dx2rho = dx2rho/(dx)**2

    if E == 0:
        dxsigmarho=0
    else:
        sigmarho = sigma(rho)
        dxsigmarho = np.zeros_like(rho)
        dxsigmarho[1:-1] = sigmarho[2:]-sigmarho[:-2]
        dxsigmarho = dxsigmarho/(2*dx)

    # return (np.gradient( D(rho)*np.gradient(rho,dx), dx ))[1:-1]
    return ( dxDrho*dxrho + Drho*dx2rho - E*dxsigmarho)[1:-1]


def rhoBar(D, rhoL=0.4, rhoR=0.6, x=None, sigma=None, E=0, verbose=False):
    """
    Calculate the steady state profile for a 1D system.
    D, sigma - Diffusion and mobility coefficients (must supply functions even if they are constant)
    rhoL, rhoR - boundary conditions.
    E - bulk field
    """
    if x is None:
        x = np.linspace(0, 1)
    rho0 = rhoL * (1-x) + rhoR * x + 0.02
    rho0[0] = rhoL
    rho0[-1] = rhoR
    dx = np.gradient(x)
    residual = lambda rho: steadyStateEquation(rho, rhoL, rhoR, D, sigma, E, dx)
    try:
        rhoBulk = opti.newton_krylov(residual, rho0[1:-1], method="gmres", x_rtol=1e-9, verbose=verbose)
    except opti.nonlin.NoConvergence:
        try:
            rhoBulk = opti.newton_krylov(residual, rho0[1:-1], method="lgmres", x_rtol=1e-9, verbose=verbose)
        except opti.nonlin.NoConvergence:
            try:
                rhoBulk = opti.anderson(residual, rho0[1:-1], x_rtol=1e-9, verbose=verbose)
            except opti.nonlin.NoConvergence:
                rhoBulk = opti.newton_krylov(residual, rho0[1:-1], method="gmres", x_rtol=1e-9, iter=15000, verbose=verbose)

    rho = rhoBulk
    rho = np.insert(rho, 0, rhoL)
    rho = np.append(rho, rhoR)
    return rho


def twoPointCorrLHS(C_Bulk, Drho0x, Drho0y, sigPrimeRho0x, sigPrimeRho0y, E, dx):
    """A helper function for calculating the 2-point correlation function"""
    C = np.zeros((len(dx),len(dx)))
    C[1:-1,1:-1] = C_Bulk
    dx2C = np.zeros((len(dx),len(dx)))
    dy2C = np.zeros((len(dx),len(dx)))
    dx2 = dx**2

    dy2C[1:-1] = (C[2:]   - 2*C[1:-1] + C[:-2])
    dy2C[0]    = (C[1]    - 2*C[0]    + 0)
    dy2C[-1]   = (0 - 2*C[-1]   + C[-2])
    dy2C = dy2C / dx2

    dx2C[:,1:-1] = (C[:,2:] - 2*C[:,1:-1] + C[:,:-2])
    dx2C[:,0]    = (C[:,1]  - 2*C[:,0]    + 0)
    dx2C[:,-1]   = (0   - 2*C[:,-1]   + C[:,-2])
    dx2C = dx2C / dx2

    dyDrho0y = np.zeros((len(dx),len(dx)))
    dyDrho0y[1:-1] = (Drho0y[2:] - Drho0y[:-2])
    dyDrho0y = dyDrho0y/(2*dx)

    dxDrho0x = np.zeros((len(dx),len(dx)))
    dxDrho0x[:,1:-1] = (Drho0x[:,2:] - Drho0x[:,:-2])
    dxDrho0x = dxDrho0x/(2*dx)

    if E == 0:
        sigPrimeRho0x, sigPrimeRho0y = 0,0
        dxsigPrimeRho0x, dysigPrimeRho0y = 0,0
    else:
        dysigPrimeRho0y = np.zeros((len(dx),len(dx)))
        dysigPrimeRho0y[1:-1] = (sigPrimeRho0y[2:] - sigPrimeRho0y[:-2])
        dysigPrimeRho0y = dysigPrimeRho0y/(2*dx)

        dxsigPrimeRho0x = np.zeros((len(dx),len(dx)))
        dxsigPrimeRho0x[:,1:-1] = (sigPrimeRho0x[:,2:] - sigPrimeRho0x[:,:-2])
        dxsigPrimeRho0x = dxsigPrimeRho0x/(2*dx)

    dx2Drho0x = np.zeros((len(dx),len(dx)))
    dy2Drho0y = np.zeros((len(dx),len(dx)))

    dy2Drho0y[1:-1] = (Drho0y[2:]   - 2*Drho0y[1:-1] + Drho0y[:-2])
    dy2Drho0y[0]    = (Drho0y[1]    - 2*Drho0y[0]    + 0)
    dy2Drho0y[-1]   = (0 - 2*Drho0y[-1]   + Drho0y[-2])
    dy2Drho0y = dy2Drho0y / dx2

    dx2Drho0x[:,1:-1] = (Drho0x[:,2:] - 2*Drho0x[:,1:-1] + Drho0x[:,:-2])
    dx2Drho0x[:,0]    = (Drho0x[:,1]  - 2*Drho0x[:,0]    + 0)
    dx2Drho0x[:,-1]   = (0   - 2*Drho0x[:,-1]   + Drho0x[:,-2])
    dx2Drho0x = dx2Drho0x / dx2

    dyC, dxC = np.gradient(C, dx[0])

    return (dx2Drho0x*C+Drho0x*dx2C+2*dxDrho0x*dxC
            + dy2Drho0y*C+Drho0y*dy2C+2*dyDrho0y*dyC
            -sigPrimeRho0x*E*dxC - dxsigPrimeRho0x*E*C
            -sigPrimeRho0y*E*dyC - dysigPrimeRho0y*E*C
            ) [1:-1, 1:-1]


def twoPointCorr(D, sigma, rhoL=0.4, rhoR=0.6, E=0, sigmaPrime=None, x=None, rho0=None, verbose=False):
    """
    Calculates the 2-point correlation function for a 1D system.
    D, sigma - Diffusion and mobility coefficients (must supply functions even if they are constant)
    sigmaPrime - Derivative of the mobility w.r.t. the density (d\sigma / d\rho)
    rhoL, rhoR - boundary conditions.
    E - bulk field
    """
    if x is None:
        x = np.linspace(0, 1)
    xMat, yMat = np.meshgrid(x,x)
    C0 = np.zeros_like(xMat)
    if rho0 == None:
        if (rhoL == rhoR):
            rho0 = rhoL*np.ones_like(x)
        else:
            rho0 = rhoBar(D, rhoL, rhoR, x, sigma=sigma, E=E)
    rho0[0] = rhoL
    rho0[-1] = rhoR
    dx = np.gradient(x)

    # Calculate right-hand side
    d2sigma = np.gradient(np.gradient(sigma(rho0), dx[0]), dx[0])
    d2sigmaMat, _ = np.meshgrid(d2sigma, d2sigma)
    if E == 0:
        dsigPrimesigOverDMat = 0
    else:
        dsigPrimesigOverD = np.gradient(sigma(rho0)*sigmaPrime(rho0)*E/D(rho0), dx[0])
        dsigPrimesigOverDMat, _ = np.meshgrid(dsigPrimesigOverD, dsigPrimesigOverD)
    # plt.figure(10)
    # plt.plot(x,rho0)
    # plt.show()
    # This is merely an approximation of a delta function!
    deltaVariance = (10 * min(dx))**2 # the smaller this value, the better the approximation
    # diracDelta = lambda x: 0.5/np.sqrt(np.pi*deltaVariance) * np.exp(-x**2/deltaVariance)
    diracDelta = lambda x: 0.5*(x==0)
    RHS = (-d2sigmaMat + dsigPrimesigOverDMat) * diracDelta(xMat-yMat)
    RHS_Bulk = RHS[1:-1,1:-1]

    # Calculate the left-hand side
    Drho0 = D(rho0)
    Drho0x, Drho0y = np.meshgrid(Drho0, Drho0)
    if E == 0:
        sigPrimeRho0x, sigPrimeRho0y = 0,0
    else:
        sigPrimeRho0 = sigmaPrime(rho0)
        sigPrimeRho0x, sigPrimeRho0y = np.meshgrid(sigPrimeRho0, sigPrimeRho0)
    residual = lambda C_Bulk: twoPointCorrLHS(C_Bulk, Drho0x, Drho0y, sigPrimeRho0x, sigPrimeRho0y, E, dx) - RHS_Bulk
    C_Bulk = opti.newton_krylov(residual, C0[1:-1,1:-1], method="gmres", verbose=verbose, x_rtol=1e-7)
    C = np.zeros((len(dx),len(dx)))
    C[1:-1,1:-1] = C_Bulk
    return C



if __name__ == "__main__":
    # Perform simple tests
    import matplotlib.pyplot as plt

    D = lambda rho: np.ones_like(rho)
    sigma = lambda rho: 2 * rho * (1 - rho)
    sigmaPrime = lambda rho: 2 * (1 - 2 * rho)
    rho0 = 0.3
    rho1 = 0.8
    L = 50
    x = np.linspace(0, 1, L)
    E1 = 4
    E2 = -3
    C1 = twoPointCorr(D=D, rhoL=rho0, rhoR=rho1, x=x, sigma=sigma, sigmaPrime=sigmaPrime, E=E1, verbose=True)
    C2 = twoPointCorr(D=D, rhoL=rho0, rhoR=rho1, x=x, sigma=sigma, sigmaPrime=sigmaPrime, E=E2, verbose=True)

    plt.figure()
    plt.pcolormesh(x, x, C1)
    plt.title("C(x,x')")

    # Take cross-sections by fixing one of the x's
    selected_index = L // 5
    plt.figure()
    plt.plot(x, C1[selected_index, :], label="E=%f" % E1)
    plt.plot(x, C2[selected_index, :], label="E=%f" % E2)
    plt.title("C(x,x') with x'={:.2}".format(x[selected_index]))
    plt.legend()

    # Plot the autocorrelation functions
    autocorr1 = [C1[i, i] for i in range(L)]
    autocorr2 = [C2[i, i] for i in range(L)]
    plt.figure()
    plt.plot(x, autocorr1, label="E=%f" % E1)
    plt.plot(x, autocorr2, label="E=%f" % E2)
    plt.title("Autocorrelation function C(x,x)")
    plt.legend()

    plt.show()
    