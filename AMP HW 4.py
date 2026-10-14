#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 18:17:20 2026

@author: ameliagandhi
"""

# imports 
import numpy as np 
import scipy
import matplotlib.pyplot as plt 
import scipy 

def linear_fit(x, b, m): 
    return b + m*x
#%%
# PROBLEM 3 
# PART A 

# figure 
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# important constants 
h = 6.626e-34
c = 3e8
kB = 1.38e-23

# wavelength, frequency, temperatures
wvlngth_nm = np.logspace(1, 7, 10000) #
wvlngth = wvlngth_nm * 10**(-9) # m 
freq = c/wvlngth

T = 6000
T_list = np.arange(5990, 6011)

# plank function at different temperatures
plank_range_w = np.array([(2*h*c**2/wvlngth**5)/(np.exp(h*c/(wvlngth*kB*T_i))-1) for T_i in T_list])
plank_range_v = np.array([(2*h*freq**3/c**2)/(np.exp(h*freq/(kB*T_i))-1) for T_i in T_list])

# loop through the plank function at different temperatures, fit to the temperatures, and find a and b 
a_list = [] 
b_list = [] 

for i in range(len(plank_range_v[0])):
    plank_fit_v = np.array(plank_range_v[:, i])
    popt_v, pcov = scipy.optimize.curve_fit(linear_fit, np.log(T_list), np.log(plank_fit_v), p0=[0,1])
    a = popt_v[1]
    a_list.append(a)

for i in range(len(plank_range_w[0])): 
    plank_fit_w = np.array(plank_range_w[:, i])
    popt_w, pcov = scipy.optimize.curve_fit(linear_fit, np.log(T_list), np.log(plank_fit_w), p0=[0,1])
    b = popt_w[1]
    b_list.append(b)

# plot a and b 
ax.plot(wvlngth_nm, a_list, c='skyblue', linewidth=4, label='a')
ax.plot(wvlngth_nm, b_list, c='blue', linestyle='-.', label='b')

# show where the limit approaches for a and b 
print('Long wavelength limits')
print('a', a_list[-1])
print('b', b_list[-1])

# find where the maximum Bv and Blambda is at T=6000 K
index_6000 = int(np.where(T_list == 6000)[0][0])

plank_v_6000 = plank_range_v[index_6000]
wavelength_plankmax_v = wvlngth_nm[plank_v_6000 == np.max(plank_v_6000)]
a_plankmax_v = np.array(a_list)[plank_v_6000 == np.max(plank_v_6000)]

plank_w_6000 = plank_range_w[index_6000]
wavelength_plankmax_w = wvlngth_nm[plank_w_6000 == np.max(plank_w_6000)]
b_plankmax_w = np.array(b_list)[plank_w_6000 == np.max(plank_w_6000)]

# plot where this wavelength is 
plt.axvline(wavelength_plankmax_w, c='gray', linewidth=4, label='wavelength with maximized ' + r'$B_{\lambda}$')
plt.axvline(wavelength_plankmax_v, c='k', linestyle='-.', label='frequency with maximized ' + r'$B_{\nu}$')

# print what a and b are 
print('Wavelength where Plank Function is Maximized')
print(r'$B_f$', wavelength_plankmax_v[0], 'a=', a_plankmax_v[0])
print(r'$B_w$', wavelength_plankmax_w[0], 'b=', b_plankmax_w[0])

# part d 
# find where 4 is 
index_a4 = int(np.where(np.abs(np.array(a_list) - 4) == np.min(np.abs(np.array(a_list) - 4)))[0][0])
plt.axvline(wvlngth_nm[index_a4], c='yellowgreen', linewidth=4, label='a=4')

index_b4 = int(np.where(np.abs(np.array(b_list) - 4) == np.min(np.abs(np.array(b_list) - 4)))[0][0])
plt.axvline(wvlngth_nm[index_b4], c='darkolivegreen', linestyle='-.', label='b=4')

print('Wavelength where a = 4 and b = 4')
print('a:', wvlngth_nm[index_a4])
print('b:', wvlngth_nm[index_b4])


# prettify 
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Constant')
ax.legend(frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')


#%%
# PROBLEM 4 

# define the figure 
fig, ax = plt.subplots(1, 1)

# part c
# define constants 
T_in = 1e6 # K 
R_in = 1e11 # m
R_out = 1e16 # m
R_mid = np.sqrt(R_in*R_out)
 
h = 6.626e-34
c = 3e8
kB = 1.38e-23

# define r, T, wavelength, frequency
r = np.linspace(1e11, 1e16, 200)
T_profile = T_in*(R_in/r)**(3/4) * (1+ r**2/R_mid)**(1/8)

wvlngth_nm = np.logspace(-1, 7, 100)
wvlngth = wvlngth_nm*1e-9
freq = c/wvlngth

# loop through each frequency to find B_v and I_v
I_v = [] 
for f in freq: 
    B_v = np.array((2*h*f**3/c**2)/(np.exp(h*f/(kB*T_profile)) -1))
    I_v_i = scipy.integrate.trapezoid(B_v * r * 2*np.pi, r) 
    I_v.append(I_v_i)

ax.plot(wvlngth_nm, I_v, c='k', label=r'$I_{\nu}$')

# part d 
# overplot B_v at R_in and R_out 
B_v_in = np.array((2*h*freq**3/c**2)/(np.exp(h*freq/(kB*T_profile[0])) -1))
B_v_out = np.array((2*h*freq**3/c**2)/(np.exp(h*freq/(kB*T_profile[-1])) -1))

ax.plot(wvlngth_nm, 2e25*B_v_in, c='lightcoral', label=r'$B_{\nu in}$') # 2e25
ax.plot(wvlngth_nm, .6e33*B_v_out, c='brown', label=r'$B_{\nu out}$') # .6e33

# plot boundary between where we should take spectral diagnostics for the inner and outer part of the disk
plt.axvline(4, c='gray', linestyle='--', label='4 nm') 

# prettify 
ax.set_ylim(1e15, 1e30)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel(r'$I_{\nu} \ \ and \ \ B_{\nu}$')
ax.legend(frameon=False)


