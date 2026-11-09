#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:47:44 2026

@author: ameliagandhi
"""

import numpy as np 
import scipy 
import matplotlib.pyplot as plt 
from scipy.special import assoc_legendre_p, genlaguerre, factorial
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import pandas as pd 

#%%

# Problem 2 
# ---------

# part b 
# ------

# define constants
a0 = 5.292*10**(-11) # bohr radius 
e = 1.602e-19 # electron charge 
ep0 = 8.8541e-12 # epsillon naught 
h = 6.626e-34 # planks constant 
c = 3e8

# make functions for R and Theta 
def Theta(l, m, theta): 
    x = np.cos(theta)
    P = assoc_legendre_p(l, m, x)
    
    # integral_top = trapezoid((assoc_legendre_p(l, 0, x))**2, x)
    # integral_bottom = trapezoid((assoc_legendre_p(l, m, x))**2, x) # [::-1]
    # normalization = np.sqrt(integral_top/integral_bottom)
    
    # # normalize everything
    # P_normalized = P[0]*normalization
    # return P_normalized
    
    normalization = np.sqrt((2*l+1)/(4*np.pi) *factorial(l-m)/factorial(l+m)) # found this normalization with AI 
    
    return P[0]*normalization

def R(n, l, r): 
    kappa = 1/(a0*n)
    normalization = np.sqrt((2*kappa)**3*factorial(n-l-1) / (2*n*factorial(n+l)))
    other_factors = np.exp(-kappa*r) * (2*kappa*r)**l
    L = genlaguerre(n-l-1, 2*l+1)(2*kappa*r)
    R = normalization*other_factors*L
    return R 


# compute Mij  
def Mij(ni, nj, li, lj, mi, mj, r, theta): 
    
    # find Ri, Rj, Thetai, Thetaj
    Ri = R(ni, li, r)
    Rj = R(nj, lj, r)
    Thetai = Theta(li, mi, theta)
    Thetaj = Theta(lj, mj, theta)
    
    # integrate 
    R_int = np.trapezoid(r**3*Ri*Rj, r)
    Theta_int = np.trapezoid(np.sin(theta)*np.cos(theta)*Thetai*Thetaj, theta)
    
    # compute Mij
    Mij = 2*np.pi/a0 * R_int * Theta_int
    
    return Mij 

# compute Aij 
def Aij(Mij, nu0): 
    factor = 16 * np.pi**3 * nu0**3 * e**2 * a0**2 / (3*ep0*h*c**3)
    return factor * Mij**2

### compute Mij for 1s -> 1s and 3d -> 3d

# define theta and r
theta = np.linspace(0, np.pi, 1000) # theta runs from 0 to pi 
r = np.linspace(0, 100*a0, 1000)

# calculate! 
M1s1s = Mij(1, 1, 0, 0, 0, 0, r, theta)
print(f'1s -> 1s: {M1s1s:.5f}')

M3d3d = Mij(3, 3, 2, 2, 0, 0, r, theta)
print(f'3d -> 3d: {M3d3d:.5f}')

# part c 
# ------

# make a list of n's 
ni_range = np.arange(1, 10)
nj_range = ni_range + 1

# define wavelength for checking with lecture notes 

Aji_list = [] 
ni_list = [] 
nj_list = [] 

# loop through the principle quantum numbers 
for i, ni in enumerate(ni_range):
    # loop through nj's larger than ni 
    for nj in nj_range[i:]: 
        # calculate nu0
        nu0 = (13.6056*e/h) * (1/ni**2 - 1/nj**2) # got help from AI 
        
        # define r 
        r = np.linspace(0, np.max([ni, nj])**2*40*a0, 1000)
        
        # compute Mji 
        Mji = Mij(nj, ni, 1, 0, 0, 0, r, theta)
        
        # calculate the einstein coefficent 
        Aji = Aij(Mji, nu0)
        Aji_list.append(Aji)
        ni_list.append(ni)
        nj_list.append(nj)
    
# make a datframe and a chart 
table = pd.DataFrame({r'$n_jp$': nj_list, r'$n_is$': ni_list, r'$A_{ji}$': Aji_list}) 

table[r'$A_{ji}$'] = table[r'$A_{ji}$'].map(lambda x: f'{x:.2e}')

# print the table 
fig, ax = plt.subplots(1,1, figsize=(3, 10))
tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
ax.set_axis_off()


# part d 
# ------

# treat ni an nj as dependent variables and Aji as a dependent variable 
# log(Aji) = log(C) - alpha log(ni) - beta log(nj) 

# define the fit at every ni 
def fit(combo, alpha, beta, logc):
    ni, nj = combo  # Unpack the independent variables
    return logc - alpha*ni - beta*nj

# perform the fit, find alpha, beta, and c (take the exponent of logc which we fitted to)
popt, pcov = curve_fit(fit, (np.log(ni_list), np.log(nj_list)), np.log(Aji_list), p0=[1, 1, 1])

alpha_best, beta_best, c_best = popt[0], popt[1], np.exp(popt[2]) 

print(f'alpha: {alpha_best:.2f}')
print(f'beta: {beta_best:.2f}')
print(f'c: {c_best:.2e}')

# now plot everything! 
fig, ax = plt.subplots(1,1)
# loop through making fits at every ni 
for ni in ni_range: 
    condition = ni_list == ni
    ni_allowed = np.array(ni_list)[condition]
    nj_allowed = np.array(nj_list)[condition]
    Aji_allowed = np.array(Aji_list)[condition]
    
    # plot the data 
    ax.scatter(nj_allowed, Aji_allowed, s=10, label=rf'$n_i$ = {ni}')
    
    # perform the fit and plot it 
    if len(ni_allowed) > 2: 
        popt, pcov = curve_fit(fit, (np.log(np.array(ni_allowed)), np.log(np.array(nj_allowed))), np.log(np.array(Aji_allowed)), p0=[1, 1, 1])
        fitted_Aji = popt[2] - popt[0]*np.log(ni_allowed) - popt[1]*np.log(nj_allowed)
        ax.plot(nj_allowed, np.exp(fitted_Aji))
    
    # prettify
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False) 
    ax.set_xlabel(r'$n_j$')
    ax.set_ylabel(r'$A_{ji}$')

#%%

# Problem 3 
# ---------

# input the table as a set of arrays (all in angstroms)
lambda_0 = np.array([977, 1032, 1175, 1216, 1335, 1394, 1548, 1640, 1808, 1909, 2056, 2175])
del_lambda_D = np.array([.3898, .3355, .3647, 1.2696, .5181, .3747, .4711, .9163, .4731, .6477, .4461, .5537])/(2*np.sqrt(2*np.log(2)))
ion_mass_au = np.array([12, 16, 12, 1, 12, 28.085, 12, 4, 28.085, 12, 51.996, 55.845]) 
ion_mass_kg = 1.66e-27*ion_mass_au


# define other constants needed 
kB = 1.38e-23 
c = 3e8

# define our x and y values for the linear fit 
y = (c*del_lambda_D/lambda_0)**2
x= 1/ion_mass_kg 

fig, ax = plt.subplots(1, 1)
ax.scatter(x, y, c='k') 
ax.set_xlabel(r'$1/m$')
ax.set_ylabel(r'$(\frac{c \Delta \lambda_D}{\lambda_0})^2$')

# perform the linear fit 
res = scipy.stats.linregress(x, y)

# get the slope and intercept, plot the fit 
m = res.slope
b = res.intercept

ax.plot(x, m*x + b, c='green', linestyle='--', label=rf'$2 k_B T$ = {m:.2e}, $\xi^2$ = {b:.2e}')
ax.legend(frameon=False) 

# find T and xi 
xi = np.sqrt(b) 
T = m/(2*kB)

print(f'xi = {xi:.2e} J/kg')
print(f'T = {T:.2e}')

#%%

# Problem 4 
# ---------
# make values for theta and v 
theta_range = np.linspace(0, 2*np.pi, 1000)
v_range = np.linspace(0, 3e8, 1000)

# make a grid of these values 
theta, v = np.meshgrid(theta_range, v_range)

# calculate v_pm and lambda_obs/lambda_0
v_pm = (v*np.sin(theta)) / (1  + v/c * np.cos(theta))
lambda_obs_0 = (1 + v*np.cos(theta)/c) / np.sqrt(1 - v**2/c**2)

v_pm_c = v_pm/c # v_pm in units of c 

# find where the indices are close to our observed data 
v_pm_obs = 2.52272 
lambda_obs_0_obs = 0.343035
rows, cols = np.where((np.abs(v_pm_c - v_pm_obs) < 0.01) & (np.abs(lambda_obs_0 - lambda_obs_0_obs) < 0.01))

# plot thetas and vs that match the observed data 
fig, ax = plt.subplots(1, 1)
ax.scatter(np.rad2deg(theta_range[cols]), v_range[rows]/c, c='k', label=r'Allowed values of $\theta$ and $v/c$')

median_theta = np.rad2deg(theta_range[int(np.median(cols))])
median_v = v_range[int(np.median(rows))]/c
plt.scatter(median_theta, median_v, c='green', s=10, label=rf'Median $\theta$ = {median_theta:.0f}$\degree$, Median $v$ = {median_v:.2f}c')

# prettify 
ax.set_xlim(np.rad2deg(np.min(theta_range)), np.rad2deg(np.max(theta_range)))
ax.set_ylim(np.min(v_range)/c, np.max(v_range)/c)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$v/c$')
ax.legend(frameon=False) 

