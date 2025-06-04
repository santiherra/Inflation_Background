import numpy as np
from scipy.integrate import odeint, quad
from scipy.optimize import root
from scipy.interpolate import interp1d

MP = 1

''' POTENTIAL '''

def critical(a):  # Critical value of b for inflection point
    return 1 + a**2*(-1 + (4.5/a**2 - 1)**(2/3))/3

def inflection(a, b):
    th = (3**.5*a*b*(a**6/27*(4.5/a**2 - 1)**2 - (b - 1 + a**2/3)**3)**.5 + (1 - b)**3 + 1.5*a**2*b)**(1/3)
    return np.real((b - 1)/(a*b) + .5/(a*b)*(th + ((b-1)**2 + a**2*b)/th))

def pot(x, a, b, v, l):  # Potential, V(x)
    return l/12*v**4*x**2*(6 - 4*a*x + 3*x**2)/(1 + b*x**2)**2  # x -> dimensionless field

def pot_d(x, a, b, v, l):  # First derivative, V'(x)
    return l*v**2*x*(a*b*x**3 + 3*(1-b)*x**2 - 3*a*x + 3)/3/(1 + b*x**2)**3

def pot_dd(x, a, b, v, l):  # Second derivative, V''(x)
    numerator = -l*v**4*(-3 + 3*(8*b - 3)*x**2 - 9*(b - 1)*b*x**4 + 2*a*x*(3 - 8*b*x**2 + b**2*x**4))
    denominator = 3*(1 + b*x**2)**4
    return numerator / denominator

''' SYSTEM OF EQUATIONS '''

def constraint(x, x_dot, a, b, v, l): # Friedman Equation (constraint)
    return ((pot(x, a, b, v, l))/3/(1 - v**2*x_dot**2/6/MP**2))**.5/MP

def systemSR(S, N, a, b, v, l):
    x = S  # Initial values of the fields for each iteration
    x_dot = - (2*epsSR(x, a, b, v, l))**.5/v # Klein - Gordon equation
    return x_dot

def system(S, N, a, b, v, l):
    x, x_dot, H = S  # Initial values of the fields for each iteration
    x_ddot = -3*x_dot + .5*x_dot**3*v**2/MP**2 - pot_d(x, a, b, v, l)/H**2 # Klein - Gordon equation
    H_dot = -.5/MP**2*H*(x_dot*v)**2  # Acceleration equation
    return [x_dot, x_ddot, H_dot]

''' HUBBLE PARAMETERS '''

def epsSR(x, a, b, v, l):  # Flow parameter, slow-roll approximation
    return MP**2/2*(pot_d(x, a, b, v, l)/pot(x, a, b, v, l)*v)**2

def eps1(x_dot):  # Flow parameter, general
    return MP**2*(x_dot)**2/2

def efoldsSR(x, xi, a, b, v, l):  # Number of e-folds, slow-roll
    integr = lambda y: 1/(2*epsSR(y, a, b, v, l))**.5
    return quad(integr, xi, x)[0]

def eta1(x_dot, N):
    grad = np.gradient(eps1(x_dot), N)
    return grad/eps1(x_dot)

def epsilonint(N, eps):
    return interp1d(N, eps)

def etaint(eta, N):
    return interp1d(N, eta)

def etaSR(x, a, b, v, l):  # Second Hubble flow parameter, slow-roll
    num = 18 -24*a*x + (9 +12*a**2 + 72*b)*x**2 - (18*a +96*a*b)*x**3 + (9+ 54*b + 40*a**2*b - 18*b**2)*x**4 + (-60*a*b +24*a*b**2)*x**5 + b*(27*(1-b) - 4*a*b)*x**6 + 6*a*b**2*x**7
    den = x**2*(6 - 4*a*x + 3*x**2)**2*(1 + b*x**2)**2
    return 8*num/den/v**2

''' INFLATION '''

def systemsolvSR(S0, d, a, b, v, l):
    solution = np.array([S0])
    Nfolds = np.array([0])
    S0_aux = S0
    i = 0
    while (epsSR(S0_aux, a, b, v, l) <=1):
        N_span = [Nfolds[i], Nfolds[i] + d]
        sol_aux = odeint(systemSR, S0_aux, N_span, args=(a, b, v, l))
        S0_aux = sol_aux[-1]
        solution = np.concatenate((solution, S0_aux), axis=0)
        Nfolds = np.concatenate((Nfolds, [N_span[-1]]), axis=0)
        i += 1
    return solution, Nfolds

def systemsolv(S0, d, a, b, v, l):
    solution = np.array([S0])
    Nfolds = np.array([0])
    S0_aux = S0
    i = 0
    while (eps1(S0_aux[1]*v) <=1):
        N_span = [Nfolds[i], Nfolds[i] + d]
        sol_aux = odeint(system, S0_aux, N_span, args=(a, b, v, l))
        S0_aux = sol_aux[-1]
        solution = np.concatenate((solution, [S0_aux]), axis=0)
        Nfolds = np.concatenate((Nfolds, [N_span[-1]]), axis=0)
        i += 1
    return solution, Nfolds

def mukhanovsolve(S, k, d, a, b, v, l):
    sol, N = systemsolv(system, S, d, a, b, v, l)
    eps = eps1(sol[:, 2]) 
    eta = eta1(sol[:, 1], N)
    aH = np.exp(N)*sol[:, 2]
    dlog = np.gradient(np.gradient(np.log(eta), N), N)
    epsint = interp1d(eps, N)
    etaint = interp1d(eta, N)
    Hc = interp1d(N, aH)
    dleta = interp1d(dlog, N)
    def mukhanov(S_p, N, k):
        u, u_dot, v, v_dot = S_p
        u_ddot = -(1-epsint(N))*u_dot + ((k/Hc(N))**2-(1+.5*etaint(N))*(2-epsint(N)+.5*etaint(N)) + .5*dleta(N))*u
        v_ddot = -(1-eps(N))*v_dot + ((k/Hc(N))**2-(1+.5*etaint(N))*(2-epsint(N)+.5*etaint(N)) + .5*dleta(N))*v
        return [u_dot, u_ddot, v_dot, v_ddot]
    
    def boundary(k,N,aH,Nic_cond,Nshs_cond):
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx, array[idx]
    
        ic_diff=k-Nic_cond*aH
        shs_diff=k-Nshs_cond*aH
        idx_ic,ah_ic = find_nearest(ic_diff,0)
        idx_shs,ah_shs = find_nearest(shs_diff,0)
        Nic = N[idx_ic]
        Nshs = N[idx_shs]
        return idx_ic,Nic,Nshs
    
    S_P = [1/(2*k)**.5, 0, 0, -(k/2)**.5]
    solution = [S_P]
    Nfolds = [boundary(k, N, aH, 100, 0.01)[1]]
    S_aux = S_P
    i = 0
    while k > .01*Hc(Nfolds[-1]):
        N_span = [Nfolds[i], Nfolds[i] + d]
        sol_aux = odeint(mukhanov, S_aux, N_span, args=(k,))
        S_aux = sol_aux[-1]
        solution.append(S_aux)
        Nfolds.append(N_span[-1])
        i += 1
    return solution, Nfolds
