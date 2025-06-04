import numpy as np
import functions_tfg as myf
import matplotlib.pyplot as plt
import scienceplots
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.style.use(['science', 'no-latex'])

MP = 1

def plot_pot(S0, x, d, a, b, v, l, font=13):
    x_val = (myf.systemsolv(S0, d, a, b, v, l)[0])[:, 0]
    x_crit = myf.inflection(a, b)
    
    fig, ax = plt.subplots(1, 1, dpi=200)
    V = myf.pot(x, a, b, v, l)
    ax.plot(x, V/(l*v**4), color='red')
    plt.axvspan(x_val[0], x_val[-1], color='lightblue', alpha=0.2)
    ax.axvline(x = x_val[0], linestyle='--', color='blue', linewidth=.5)
    ax.axvline(x = x_val[-1], linestyle='--', color='blue', linewidth=.5)
    ax.axvline(x = x_crit, color='gray', linewidth=.5)
    ax.set_xlabel(r'$\phi/v$', fontsize=font)
    ax.set_ylabel(r'$V(\phi)/(\lambda v^2)$', fontsize=font)
    return fig, ax

def plot_epsilon(S0, x_dN, d, a, b, v, l, font=13):
    vals, Nfolds = myf.systemsolv(S0, d, a, b, v, l)
    NfoldsSR = np.linspace(0, 80, 500)
    valsSR, NfoldsSR = myf.systemsolvSR(S0[0], d, a, b, v, l)
    x_vals_SR = valsSR
    x_vals, x_dot_vals = np.array(vals)[:, 0], np.array(vals)[:, 1]
    epsilonSR = myf.epsSR(x_vals_SR, a, b, v, l)
    epsilonexact = myf.eps1(x_dot_vals*v)
    etaabsSR = np.abs(myf.etaSR(x_vals_SR*v, a, b, v, l))
    etaexact = np.abs(myf.eta1(x_dot_vals, Nfolds))
    epsintSR = myf.epsilonint(NfoldsSR, epsilonSR)
    etaintSR = myf.epsilonint(NfoldsSR, etaabsSR)
    epsint = myf.epsilonint(Nfolds, epsilonexact)
    etaint = myf.epsilonint(Nfolds, etaexact)
    Nf = np.linspace(0, Nfolds[-1], 10000)
    NfSR = np.linspace(0, NfoldsSR[-1], 10000)
    ind = [epsint(Nf)<=1]
    ind2 = [epsintSR(NfSR)<=1]
    NfoldsSR_zoom = np.linspace(70, NfoldsSR[-1], 100000)
    epsSR_zoom = epsintSR(NfoldsSR_zoom)
    ind3 = [epsSR_zoom<=1]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    fig.tight_layout(pad=6.0)
    ax[0].set_yscale('log') 
    ax[0].axhline(y=1, color='gray', linewidth=.5)
    ax[0].plot((NfSR)[ind2[0]], epsintSR(NfSR)[ind2[0]], '-', color='tomato', label='Slow-Roll')
    ax[0].plot((Nf)[ind[0]], epsint(Nf)[ind[0]], color='cornflowerblue', label='Exact')
    ax[0].set_xlabel(r'$N$', fontsize=font)
    ax[0].set_ylabel(r'$\epsilon(N)$', fontsize=font)
    ax[0].legend(loc='lower left', fontsize=font)
    ax[0].tick_params(axis='both', labelsize=font)
    ax[1].set_yscale('log') 
    ax[1].plot(NfSR[ind2[0]], etaintSR(NfSR)[ind2[0]], '-', color='yellowgreen', label='Slow-Roll')
    ax[1].plot(Nf[ind[0]], etaint(Nf)[ind[0]], color='teal', label='Exact')
    ax[1].set_xlabel(r'$N$', fontsize=font)
    ax[1].set_ylabel(r'$|\eta|(N)$', fontsize=font)
    ax[1].legend(loc='lower right', fontsize=font)
    ax[1].tick_params(axis='both', labelsize=font)

    return fig, ax

def plot_efolds(S0, x_dN, d, a, b, v, l, font=13):
    vals, Nfolds = myf.systemsolv(S0, d, a, b, v, l)
    x_vals_SR, NfoldsSR = myf.systemsolvSR(S0[0], d, a, b, v, l)
    x_vals, x_dot_vals = vals[:, 0], vals[:, 1]
    efoldsSRv = np.vectorize(myf.efoldsSR)
    print(x_vals[0])
    
    xf2 = np.linspace(1e-4, S0[0], 500)  # Interval for N plot
    mask = (x_vals > x_dN[0]) & (x_vals < x_dN[-1])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    fig.tight_layout(pad=6.0)
    ax[0].axhline(y=myf.inflection(a, b), color='gray', linewidth=.5)
    ax[0].plot(NfoldsSR, x_vals_SR, color='green', label='Slow-Roll')
    ax[0].plot(Nfolds, x_vals, color='navy', label='Exact')
    ax[0].set_xlabel(r'$N$ ', fontsize=font)
    ax[0].set_ylabel(r'$\phi/v$ ', fontsize=font)
    ax[0].legend(fontsize=font)
    ax[0].tick_params(axis='both', labelsize=font)
    ax[1].axvline(x=myf.inflection(a, b), color='gray', linewidth=.5)
    ax[1].plot(x_dN, (1/(2*myf.epsSR(x_dN, a, b, v, l))**.5*v), color='limegreen', label='Slow-Roll')
    ax[1].plot(x_vals[mask], (-1/x_dot_vals[mask]), color='purple', label='Exact')
    ax[1].set_xlabel(r'$\phi/v$ ', fontsize=font)
    ax[1].set_ylabel(r'$ dN/d\phi$ ', fontsize=font)
    ax[1].legend(loc='upper left', fontsize=font)
    ax[1].tick_params(axis='both', labelsize=font)

    return fig, ax

def plot_ns(S0, d, a, b, v, l, font=13):
    vals, Nfolds = myf.systemsolv(S0, d, a, b, v, l)
    NfoldsSR = np.linspace(0, 80, 500)
    valsSR, NfoldsSR = myf.systemsolvSR(S0[0], d, a, b, v, l)
    x_vals_SR = valsSR
    x_vals, x_dot_vals = np.array(vals)[:, 0], np.array(vals)[:, 1]
    epsilonSR = myf.epsSR(x_vals_SR, a, b, v, l)
    epsilonexact = myf.eps1(x_dot_vals*v)
    etaabsSR = np.abs(myf.etaSR(x_vals_SR*v, a, b, v, l))
    etaexact = np.abs(myf.eta1(x_dot_vals, Nfolds))
    epsintSR = myf.epsilonint(NfoldsSR, epsilonSR)
    etaintSR = myf.epsilonint(NfoldsSR, etaabsSR)
    epsint = myf.epsilonint(Nfolds, epsilonexact)
    etaint = myf.epsilonint(Nfolds, etaexact) 

    ns = 1 - 2*epsint(Nfolds) - etaint(Nfolds)
    r = 16*epsint(Nfolds)
    print('ns at CMB: ' + str((ns)[0])) # [ind]
    print('r at CMB: ' + str((r)[0])) # [ind]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=200)
    fig.tight_layout(pad=6.0)
    ax.plot(Nfolds, (ns), '--', color='green', label='Slow-Roll')
    ax.axhline(y=1, color='gray', linewidth=.5)
    ax.set_xlabel(r'$N$ ', fontsize=font)
    ax.set_ylabel(r'$\phi/v$ ', fontsize=font)
    ax.legend(fontsize=font)
    ax.tick_params(axis='both', labelsize=font)
    return fig, ax

beta = 1

a0 = 1
b0 = myf.critical(a0) - beta*1e-4
v0 = 0.108**.5*MP
l0 = 3e-7

N_eval = np.linspace(0, 60, 1000)

x_0 =  9.27   # Initial inflaton field value
x_dot_0 = -1.413e-1  # Initial velocity of inflaton field
H_0 = myf.constraint(x_0, x_dot_0, a0, b0, v0, l0)

S0 = [x_0, x_dot_0, H_0]
S2 = [x_0]

plot_pot(S0, np.linspace(-1, 11, 400), N_eval[1]-N_eval[0], a0, b0, v0, l0)
plot_epsilon(S0, np.linspace(1.05, 1.3, 500), N_eval[1]-N_eval[0], a0, b0, v0, l0)
plot_efolds(S0, np.linspace(1.05, 1.3, 500), N_eval[1]-N_eval[0], a0, b0, v0, l0)
plot_ns(S0, N_eval[1]-N_eval[0], a0, b0, v0, l0)

plt.show()
