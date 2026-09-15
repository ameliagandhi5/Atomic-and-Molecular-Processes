#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:18:26 2026

@author: ameliagandhi
"""

# imports 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors 
import pandas as pd 
from scipy.optimize import curve_fit
from scipy.special import assoc_legendre_p, genlaguerre, factorial
from scipy.integrate import trapezoid

def power_fit(x, a, b):
    fit = a*x**b
    return fit 

#%%
#############
# PROBLEM 1 #
#############

# part a) 

# read in dataframe 
# ionization potential is in eV
ioniz_df = pd.read_csv('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/ioniz_potentials.txt', 
                       sep='\s+', skiprows=5, names=['Z', 'z', 'Ionization Potential', '_', 'Element', 'Num']) 

fig, ax = plt.subplots(1,1)

# plot all the points
ax.scatter(ioniz_df['Z'], ioniz_df['Ionization Potential'], c=ioniz_df['z']/ioniz_df['Z'], s=1, cmap='rainbow')
ax.set_yscale('log')
ax.set_xlabel('Atomic Number (Z)')
ax.set_ylabel('Ionization Potential (eV)')

# plot the gray curve 
neutral_df = ioniz_df[(ioniz_df['z'] == 0)]
ax.plot(neutral_df['Z'], neutral_df['Ionization Potential'], c='gray', linewidth=1)
ax.text(.15, .02, 'neutral atoms', transform=ax.transAxes, c='gray')

# plot the black curve 
h_like_df = ioniz_df[(ioniz_df['Z'] == ioniz_df['z'] + 1)]
ax.plot(h_like_df['Z'], h_like_df['Ionization Potential'], c='k', linewidth=1)
ax.text(.08, .58, 'Hydrogen-like ions', transform=ax.transAxes, rotation=30)

# part b) 

# plot the noble gases ('brown')
noble_df = neutral_df[(ioniz_df['Element'] == 'He') | (neutral_df['Element'] == 'Ne') 
                      | (neutral_df['Element'] == 'Ar') | (neutral_df['Element'] == 'Kr')
                      | (neutral_df['Element'] == 'Xe') | (neutral_df['Element'] == 'Rn')
                      | (neutral_df['Element'] == 'Og')]
ax.scatter(noble_df['Z'], noble_df['Ionization Potential'], marker='s', c='saddlebrown', s=10, zorder=2, label='Neutral Noble Gases')
ax.legend(frameon=False)
plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_ionization_potential_nobles.png')

#%%
# part c) 
# isolate the h-like ions, already done in h_like_df 
# plot 
fig, ax = plt.subplots(1,1)
ax.scatter(h_like_df['Z'], h_like_df['Ionization Potential'], c='k', s=1, label='Observed I for\nH-like ions')
ax.set_yscale('log')
ax.set_xlabel('Atomic Number (Z)')
ax.set_ylabel('Ionization Potential (eV)')

# plot the ionization potential fit 
fit = 13.6*h_like_df['Z']**2
ax.plot(h_like_df['Z'], fit, c='green', linewidth=1)
ax.text(.3, .6, r'$I = (13.6 eV) Z^2$', transform=ax.transAxes, c='green')
ax.legend(frameon=False)

# calculate the percent error 
h_like_perr = (fit - h_like_df['Ionization Potential'])/h_like_df['Ionization Potential']*100
ax2 = ax.twinx()
ax2.plot(h_like_df['Z'], np.abs(h_like_perr), c='blue')
ax2.set_ylabel('Percent Error (%)', c='blue')
ax2.tick_params(axis='y', color='blue', labelcolor='blue')

# report the mean of the percent error 
print(f'Mean of the Absolute Value of the Percent Error: {np.mean(np.abs(h_like_perr)):.2f}%')
plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_hlike.png')


#%%
# part d) 
# isolate the Li-like ions
Li_like_df = ioniz_df[(ioniz_df['Z'] == ioniz_df['z'] + 3)]

# plot
fig, ax = plt.subplots(1,1)
ax.scatter(Li_like_df['Z'], Li_like_df['Ionization Potential'], c='k', s=1, label='Observed Li-like ions')
ax.set_yscale('log')
ax.set_xlabel('Atomic Number (Z)')
ax.set_ylabel('Ionization Potential (eV)')
ax.legend(frameon=False) 

# fit the curve (looks like an exponential, so use that)
Li_popt, Li_pcov = curve_fit(power_fit, Li_like_df['Z'], Li_like_df['Ionization Potential'], p0=[1, 1])
Li_a = Li_popt[0]; Li_b = Li_popt[1] # get all the parameters we want 

# plot the fit 
fit = Li_a*Li_like_df['Z']**Li_b
ax.plot(Li_like_df['Z'], fit, c='green')
ax.text(.3, .6, f'I = ({Li_a:.2f})Z^{Li_b:.2f}', transform=ax.transAxes, c='green')

# plot the percent error 
Li_like_perr = (fit - Li_like_df['Ionization Potential'])/Li_like_df['Ionization Potential']*100
ax2 = ax.twinx()
ax2.plot(Li_like_df['Z'], np.abs(Li_like_perr), c='blue')
ax2.set_ylabel('Percent Error (%)', c='blue')
ax2.tick_params(axis='y', color='blue', labelcolor='blue')

# report the mean of the percent error
print(f'Mean of the Absolute Value of the Percent Error: {np.mean(np.abs(Li_like_perr)):.2f}%')
plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_Lilike.png')

#%%
# part e) 
# isolate the Na-like ions
Na_like_df = ioniz_df[(ioniz_df['Z'] == ioniz_df['z'] + 11)]

# plot
fig, ax = plt.subplots(1,1)
ax.scatter(Na_like_df['Z'], Na_like_df['Ionization Potential'], c='k', s=1, label='Observed Na-like ions')
ax.set_yscale('log')
ax.set_xlabel('Atomic Number (Z)')
ax.set_ylabel('Ionization Potential (eV)')
ax.legend(frameon=False) 

# fit the curve (looks like an exponential, so use that)
Na_popt, Na_pcov = curve_fit(power_fit, Na_like_df['Z'], Na_like_df['Ionization Potential'], p0=[1, 1])
Na_a = Na_popt[0]; Na_b = Na_popt[1] # get all the parameters we want 

# plot the fit 
fit = Na_a*Na_like_df['Z']**Na_b
ax.plot(Na_like_df['Z'], fit, c='green')
ax.text(.3, .6, f'I = ({Na_a:.2f})Z^{Na_b:.2f}', transform=ax.transAxes, c='green')

# plot the percent error 
Na_like_perr = (fit - Na_like_df['Ionization Potential'])/Na_like_df['Ionization Potential']*100
ax2 = ax.twinx()
ax2.plot(Na_like_df['Z'], np.abs(Na_like_perr), c='blue')
ax2.set_ylabel('Percent Error (%)', c='blue')
ax2.tick_params(axis='y', color='blue', labelcolor='blue')

# report the mean of the percent error
print(f'Mean of the Absolute Value of the Percent Error: {np.mean(np.abs(Na_like_perr)):.2f}%')
plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_Nalike.png')

#%%
# isolate the K-like ions
K_like_df = ioniz_df[(ioniz_df['Z'] == ioniz_df['z'] + 19)]

# plot
fig, ax = plt.subplots(1,1)
ax.scatter(K_like_df['Z'], K_like_df['Ionization Potential'], c='k', s=1, label='Observed K-like ions')
ax.set_yscale('log')
ax.set_xlabel('Atomic Number (Z)')
ax.set_ylabel('Ionization Potential (eV)')
ax.legend(frameon=False) 

# fit the curve (looks like an exponential, so use that)
K_popt, K_pcov = curve_fit(power_fit, K_like_df['Z'], K_like_df['Ionization Potential'], p0=[1, 1])
K_a = K_popt[0]; K_b = K_popt[1] # get all the parameters we want 

# plot the fit 
fit = K_a*K_like_df['Z']**K_b
ax.plot(K_like_df['Z'], fit, c='green')
ax.text(.3, .6, f'I = ({K_a:.2f})Z^{K_b:.2f}', transform=ax.transAxes, c='green')

# plot the percent error 
K_like_perr = (fit - K_like_df['Ionization Potential'])/K_like_df['Ionization Potential']*100
ax2 = ax.twinx()
ax2.plot(K_like_df['Z'], np.abs(K_like_perr), c='blue')
ax2.set_ylabel('Percent Error (%)', c='blue')
ax2.tick_params(axis='y', color='blue', labelcolor='blue')

# report the mean of the percent error
print(f'Mean of the Absolute Value of the Percent Error: {np.mean(np.abs(K_like_perr)):.2f}%')
plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_Klike.png')

#%%
# check 
# isolate the Cu-like ions
Cu_like_df = ioniz_df[(ioniz_df['Z'] == ioniz_df['z'] + 29)]

# plot
fig, ax = plt.subplots(1,1)
ax.scatter(Cu_like_df['Z'], Cu_like_df['Ionization Potential'], c='k', s=1, label='Observed Cu-like ions')
ax.set_yscale('log')
ax.set_xlabel('Atomic Number (Z)')
ax.set_ylabel('Ionization Potential (eV)')
ax.legend(frameon=False) 

# fit the curve (looks like an exponential, so use that)
Cu_popt, Cu_pcov = curve_fit(power_fit, Cu_like_df['Z'], Cu_like_df['Ionization Potential'], p0=[1, 1])
Cu_a = Cu_popt[0]; Cu_b = Cu_popt[1] # get all the parameters we want 

# plot the fit 
fit = Cu_a*Cu_like_df['Z']**Cu_b
ax.plot(Cu_like_df['Z'], fit, c='green')
ax.text(.3, .6, f'I = ({Cu_a:.2f})Z^{Cu_b:.2f}', transform=ax.transAxes, c='green')

# plot the percent error 
Cu_like_perr = (fit - Cu_like_df['Ionization Potential'])/Cu_like_df['Ionization Potential']*100
ax2 = ax.twinx()
ax2.plot(Cu_like_df['Z'], np.abs(Cu_like_perr), c='blue')
ax2.set_ylabel('Percent Error (%)', c='blue')
ax2.tick_params(axis='y', color='blue', labelcolor='blue')

# report the mean of the percent error
print(f'Mean of the Absolute Value of the Percent Error: {np.mean(np.abs(Cu_like_perr)):.2f}%')
plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_Culike.png')


#%%
#############
# PROBLEM 2 #
#############

# part a) 
# define the free parameter and independent variable 
k0_a = np.logspace(-2, 2, 5)
e = np.logspace(-5, 5, 1000)

# write the transmission coefficent 
def transmission_coeff(e, k0_a): 
    # e is an array, k0_a is a constant 
    
    S_list = np.array([])
    
    for e_i in e: 
        factor = k0_a*np.sqrt(np.abs(1-e_i))
        
        if e_i < 1 and e_i > 0: 
            S = np.sinh(factor)**2
        if e_i > 1: 
            S = np.sin(factor)**2
        S_list = np.append(S_list, S)
        
    fraction = S_list/(4*e*np.abs(1-e))
    T = (1 + fraction)**-1
    
    return T

# make a colormap for the values of k0_a
cmap = plt.cm.plasma
norm = colors.LogNorm(vmin=.01, vmax=100)

# plot everything 
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, .1], 'wspace': .1})

for k in k0_a: 
    ax[0].plot(e, transmission_coeff(e, k), color=cmap(norm(k)), linewidth=1)
    ax[0].set_xscale('log')
    
# make the colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=ax[1], label=r'$k_0a$')

# make the plot look pretty 
ax[0].set_xlabel(r'ξ = $\frac{E}{U_0}$')
ax[0].set_ylabel(r'Transmission Coefficent $T$')

plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_trans_coeff.png')

# part c) 
# overplot with the transmission coefficent for fusion 
T_fusion = np.exp(-1/e)
ax[0].plot(e, T_fusion, c='k', linestyle='--', label='T for fusion')
ax[0].plot(e, transmission_coeff(e,2.4), c='k', label=r'$k_0a \approx 2.4$')
ax[0].legend(frameon=False)

plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_trans_coeff_fusion.png')

#%%
# part b) 
# define k0_a and the function 
k0_a = np.logspace(-2, 2, 100)

def transmission_coeff_lim(k0_a): 
    return (1+k0_a**2/4)**-1

# plot everything 
fig, ax = plt.subplots(1, 1)

ax.plot(k0_a, transmission_coeff_lim(k0_a), color='k')

# make the plot look pretty 
ax.set_xlabel(r'$k_0a$')
ax.set_ylabel(r'Transmission Coefficent $T$ as $\xi \rightarrow 1$')

plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_trans_coeff_approx.png')

#%%
#############
# PROBLEM 3 #
#############

# part a) 
# make a figure 
fig, ax = plt.subplots(1,1, figsize=(6, 5))

# define the independent variables and free parameters 
x = np.linspace(-1, 1, 100)
l_list = np.array([0,1,2,3,4])
m_list = np.array([0,1,2,3,4])

l_colors = ['blue', 'blueviolet', 'deeppink', 'turquoise', 'cornflowerblue']
m_linestyle = ['-', (0, (10, 5)), (0, (5, 5)), (0, (2, 2)), ':', ]

# # make the legendre polynomials 
for l, l_c in zip(l_list, l_colors): 
    for m, m_s in zip(m_list, m_linestyle):
        
        # check this condition to make sure this is a valid combo of quantum numbers
        if l >= m:
            
            # get the legendre polynomial function 
            P = assoc_legendre_p(l, m, x)
            
            # compute the normalization factor, use trapzoidal method from scipy 
            integral_top = trapezoid((assoc_legendre_p(l, 0, x))**2, x)
            integral_bottom = trapezoid((assoc_legendre_p(l, m, x))**2, x)
            normalization = np.sqrt(integral_top/integral_bottom)
            
            # normalize everything and plot 
            P_normalized = P[0]*normalization
            ax.plot(x, P_normalized, c=l_c, linestyle=m_s, label=f'l={l}, m={m}')

# make it pretty 
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_yticklabels([])
ax.set_xlabel(r'$x = cos(\theta)$')
ax.set_ylabel(r'$P_\ell^m(\theta)$')
fig.legend(loc='center left', bbox_to_anchor=(.8,0.5), frameon=False)

fig.subplots_adjust(left=.15, right=.8, bottom=0.15, top=.99)

plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_legendre.png')

 #%%
# part b) 

# create figure 
fig, ax = plt.subplots(6, 1, figsize=(5, 11), gridspec_kw={'hspace': .2, 'height_ratios': [1, .5, .5, .3, .3, .3]})

# define variables 
n_list = [1, 2, 2, 3, 3, 3]
l_list = [0, 0, 1, 0, 1, 2]

a0 = 5.29*10**(-11)

r = np.linspace(0, 25*a0, 100)

for n, l, a in zip(n_list, l_list, ax): 
    kappa = 1/(a0*n)
    
    normalization = np.sqrt((2*kappa)**3*factorial(n-l-1) / (2*n*factorial(n+l)))
    other_factors = np.exp(-kappa*r) * (2*kappa*r)**l
    
    L = genlaguerre(n-l-1, 2*l+1)(2*kappa*r)
    
    R = normalization*other_factors*L
    
    rad_prob_density = np.abs(R)**2 * r**2
    
    # plot the curves 
    a.plot(r/a0, rad_prob_density * a0, c='k')
    a.fill_between(r/a0, rad_prob_density*a0, y2=0, color='yellowgreen')
    
    # plot vertical lines for max values 
    if a == ax[0] or a == ax[2] or a == ax[5]:
        a.axvline(r[np.argmax(rad_prob_density*a0)]/a0, linestyle='--', c='k')
    
    # make things pretty
    if a == ax[0]: 
        a.set_ylim(0, .6)
    elif a == ax[1] or a == ax[2]: 
        a.set_ylim(0, .2)
    else: 
        a.set_ylim(0, .15)
        
    # add labels 
    a.text(.8, .5, f'n = {n} \nl = {l}', transform=a.transAxes, va='center', ha='center')

    a.label_outer()
    
    
# make it pretty 
ax[-1].set_xlabel(r'$r/a_0$' + r'  ($a_0$: Bohr Radius)')
ax[0].set_title('Hydrogrem Atom: Radial Proabbility Density\n' + r'$r^2 |R_{n\ell}(r)|^2$');

plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_laguerre.png')

#%%
# part c 

# define the bohr radius 
a0 = 5.29*10**(-11)

# define analytic solution for the two methods of finding r expected 
def r_exp_analytic(n, l): 
    r = a0/2 * (3*n**2 - l*(l+1))
    return r
    
# make a function for the numerical solution
def r_exp_numerical(n, l): 
    kappa = 1/(a0*n)
    
    r = np.linspace(0, 10000*a0, 1000000)
    
    normalization = np.sqrt((2*kappa)**3*factorial(n-l-1) / (2*n*factorial(n+l)))
    other_factors = np.exp(-kappa*r) * (2*kappa*r)**l
    
    L = genlaguerre(n-l-1, 2*l+1)(2*kappa*r)
    
    R = normalization*other_factors*L
    
    rad_prob_density = np.abs(R)**2 * r**2
    
    # trapezoidal method for integration 
    rad_expectation_integral = trapezoid(rad_prob_density*r, r)
    
    return rad_expectation_integral

# make a list of n and l values to run the functions for 
# l cannot be bigger than n-1 

# make a list to store everything 
n_list = []
l_list = []
r_a_list = [] 
r_n_list = [] 

for n in range(0, 21): 
    for l in range(0, n): 
        
        # run the functions 
        r_a = r_exp_analytic(n, l)
        r_n = r_exp_numerical(n, l)
        
        # append everything
        n_list.append(n)
        l_list.append(l)
        r_a_list.append(r_a/a0)
        r_n_list.append(r_n/a0)
        
# put everything in a dataframe
table = pd.DataFrame({
    'n': n_list, 'l': l_list,
    'Analytical <r>\n(Bohr Radii)': r_a_list,
    'Numerical <r>\n(Bohr Radii)': r_n_list})

table['Analytical <r>\n(Bohr Radii)'] = table['Analytical <r>\n(Bohr Radii)'].map(lambda x: f'{x:.2f}')
table['Numerical <r>\n(Bohr Radii)'] = table['Numerical <r>\n(Bohr Radii)'].map(lambda x: f'{x:.2f}')

table.to_latex('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_r_exp_table.tex',index=False,escape=False, longtable=True)


#%%
#############
# PROBLEM 4 #
#############

# part d) 

# make a of l_i and l_f 
li_list = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40])
lf_list = np.array([0, 1, 2, 3, 4, 9, 19, 29, 39])

# define necessary constants 
m_1 = 1.99e-26 # kg
m_2 = 2.66e-26 # kg 
D = 1.13e-10 # m 
hbar = 1.05e-34 # Js
c = 3e8 # m/s
# make a function for finding the emission 

def lambda_transition(li, lf): 

    # find delta E 
    factor = hbar**2/2 * (m_1+m_2)/(m_1*m_2*D**2)
    difference_l_factor =  li*(li+1) - lf*(lf+1)
    delE = factor*difference_l_factor
    
    # find the wavelength 
    w = hbar*2*np.pi*c/delE
    return w

CO_transition_wvngths = lambda_transition(li_list, lf_list)
CO_transition_wvngths_mm = CO_transition_wvngths*1000

# define the observed wavelengths 
observed_w = [2.600758, 1.300404, 0.8669634, 0.6502515, 0.5202310, 0.2602398, 0.1303689, 0.08719042, 0.06568626]

# make a dataframe 
CO_table = pd.DataFrame({
    'Initial l': li_list,
    'Final l': lf_list,
    r'$\lambda_{calculated}$ (mm)': CO_transition_wvngths_mm,
    r'$\lambda_{obs}$ (mm)': observed_w})

CO_table[r'$\lambda_{calculated}$ (mm)'] = CO_table[r'$\lambda_{calculated}$ (mm)'].map(lambda x: f'{x:.8f}')

CO_table.to_latex('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/AMP/HW Figs/hw2_CO_table.tex',index=False,escape=False, longtable=True)
