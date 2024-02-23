# 4602B_Proj2.py Shane and Dakota

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.constants import c

c *= 10**(-3)  # Use units of km/s as in paper

# 1999 - Table 1 (high redshift)
data1999tab1 = np.genfromtxt('data1999tab1.txt', dtype=[('names', 'U8'),
                                            ('redshift', 'f8'), ('redshifterr', 'f8'),
                                            ('appmageff', 'f8'),
                                            ('appmagefferr', 'f8')])
# 1999 - Table 2 (Low redshift)
data1999tab2 = np.genfromtxt('data1999tab2.txt', dtype=[('names', 'U8'),
                                            ('redshift', 'f8'), ('redshifterr', 'f8'),
                                            ('appmagcorr', 'f8'),
                                            ('appmagcorrerr', 'f8')])
#1997 - (high redshift - all stars appear to be from table 1)
data1997 = np.genfromtxt('data1997.txt', dtype=[('names', 'U8'),
                                            ('redshift', 'f8'),
                                            ('appmag', 'f8'),
                                            ('appmagerr', 'f8')])
# Combined Data (Table 1 & 2 ) for later parts 
All_Redshift = np.concatenate((data1999tab1['redshift'], data1999tab2['redshift']))
All_Redshift_ER = np.concatenate((data1999tab1['redshifterr'], data1999tab2['redshifterr']))
All_appMAG = np.concatenate((data1999tab1['appmageff'], data1999tab2['appmagcorr']))
All_appMAG_ER = np.concatenate((data1999tab1['appmagefferr'], data1999tab2['appmagcorrerr']))

# Note script M subscript B is called the modified apparent magnitude

# QUESTION (1): Shane 

# Commented out code is to switch between 1997 data or Table 2 for the curve fit process

def appmag(z, mod_appmag):
    '''Equation (2) from assignment'''
    return mod_appmag + 5*np.log10(c*z)

# sol = curve_fit(appmag, data1997['redshift'], data1997['appmag'], sigma = data1997['appmagerr'], absolute_sigma = True)
# mod_appmag = sol[0][0]
sol = curve_fit(appmag, data1999tab2['redshift'], data1999tab2['appmagcorr'],sigma=data1999tab2['appmagcorrerr'], absolute_sigma=True)
mod_appmag = sol[0][0]
fig, ax = plt.subplots()
ax.set_title("Curve fit of Apparent Magnitude")
ax.set_xlabel("Redshift")
ax.set_ylabel("Apparent magnitude (B-band)")
ax.scatter(data1999tab2['redshift'],
           data1999tab2['appmagcorr'],
           label='Perlmutter et. al (1997)',
           c='b')
# ax.scatter(data1997['redshift'],
#            data1997['appmag'],
#            label='Perlmutter et. al (1997)',
#            c='b')
# ax.plot(np.sort(data1997['redshift']),
#         appmag(np.sort(data1997['redshift']), mod_appmag),
#         label='$m_B(z)$ for $\mathscr{M}_B=%.2f$' % mod_appmag,
#         c='g')
ax.plot(np.sort(data1999tab2['redshift']),
        appmag(np.sort(data1999tab2['redshift']), mod_appmag),
        label='$m_B(z)$ for $\mathscr{M}_B=%.2f$' % mod_appmag,
        c='g')
plt.legend()
plt.show()

# QUESTION (2): Shane and Dakota 

# Function to define the integrand in the DL (lumminosity distance) formula
def integrand(z_prime, Omega_M0, Omega_Lambda0):
    Ez = np.sqrt((Omega_M0) * ((1 + z_prime)**3) + (Omega_Lambda0))
    return 1 / Ez

# Function to calculate DL ≡ d_L * H0 (the weird looking DL)
def luminosity_distance(z, Omega_M0, Omega_Lambda0):
    # Check if z is a scalar or an array
    if np.isscalar(z):
        # If z is a scalar, directly compute and return its luminosity distance
        integral, _ = quad(integrand, 0, z, args=(Omega_M0, Omega_Lambda0))
        return ((1 + z) * c )*(integral) 
    else:
        # If z is an array, initialize an array to hold the DL values
        DL_array = np.zeros(len(z))
        for i, z_val in enumerate(z):
            # Compute the luminosity distance for each z value
            integral,_ = quad(integrand, 0, z_val, args=(Omega_M0, Omega_Lambda0))
            DL_array[i] = ((1 + z_val) * c ) * (integral)
        return DL_array

# Apparent magnitude model mB(z) using equation (1)
def mB_model(z, Omega_M0, Omega_Lambda0):
    DL = luminosity_distance(z, Omega_M0, Omega_Lambda0)
    return mod_appmag + 5 * np.log10(DL)

# Results of 1999 Table 2 data (low redshift)
params, cov = curve_fit(mB_model, data1999tab2['redshift'], data1999tab2['appmagcorr'])

# Extract fitted parameters
Omega_M0_fitted, Omega_Lambda0_fitted = params
sum = Omega_Lambda0_fitted + Omega_Lambda0_fitted
print(f"Matter density (Omega_M0): {Omega_M0_fitted} \nDark energy density (Omega_Lambda0): {Omega_Lambda0_fitted}")
print(f"Flat Universe Comparison (1999, Table 2, Low z): {sum}")

if sum < 1 and Omega_Lambda0_fitted > Omega_M0_fitted:
    print('The universe has negative curvature and will expand forever')
if sum == 1 and Omega_Lambda0_fitted > Omega_M0_fitted:
    print('The universe is flat and will expand forever')
if sum > 1 and Omega_Lambda0_fitted < Omega_M0_fitted:
    print('The universe has positive curvature and will end in a big crunch')
if sum > 1 and Omega_Lambda0_fitted > Omega_M0_fitted:
    print('The universal expansion is most definetely not slowing down')
else:
    print('Something may be wrong')

# # Results of 1999 Table 1 data (high redshift)
params_1, cov_1 = curve_fit(mB_model, data1999tab1['redshift'], data1999tab1['appmageff'])

# Extract fitted parameters
Omega_M0_fitted_1, Omega_Lambda0_fitted_1 = params_1
sum = Omega_Lambda0_fitted_1 + Omega_Lambda0_fitted_1
print(f"Matter density (Omega_M0): {Omega_M0_fitted_1} \nDark energy density (Omega_Lambda0): {Omega_Lambda0_fitted_1}")
print(f"Flat Universe Comparison (1999, Table 1, High z): {sum}")

if sum < 1 and Omega_Lambda0_fitted > Omega_M0_fitted:
    print('The universe has negative curvature and will expand forever')
if sum == 1 and Omega_Lambda0_fitted > Omega_M0_fitted:
    print('The universe is flat and will expand forever')
if sum > 1 and Omega_Lambda0_fitted < Omega_M0_fitted:
    print('The universe has positive curvature and will end in a big crunch')
if sum > 1 and Omega_Lambda0_fitted > Omega_M0_fitted:
    print('The universal expansion is most definetely not slowing down')
else:
    print('Something may be wrong')

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# Plot for 1999 Table 2
axs[0].scatter(data1999tab2['redshift'], data1999tab2['appmagcorr'], label='1999 Table 2', c='blue')
axs[0].errorbar(data1999tab2['redshift'],data1999tab2['appmagcorr'], xerr=data1999tab2['redshifterr'], yerr=data1999tab2['appmagcorrerr'], fmt='o', capsize=5, label='Error Bars')
axs[0].plot(np.sort(data1999tab2['redshift']), appmag(np.sort(data1999tab2['redshift']), mod_appmag), label='Fit Table 2', c='green')
axs[0].set_title('1999 Table 2 Data and Fit')
axs[0].set_xlabel('Redshift')
axs[0].set_ylabel('Apparent magnitude (B-band)')
axs[0].legend()
# Plot for 1999 Table 1
axs[1].scatter(data1999tab1['redshift'], data1999tab1['appmageff'], label='1999 Table 1', c='blue')
axs[1].errorbar(data1999tab1['redshift'],data1999tab1['appmageff'], xerr=data1999tab1['redshifterr'], yerr=data1999tab1['appmagefferr'], fmt='o', capsize=5, label='Error Bars')
axs[1].plot(np.sort(data1999tab1['redshift']), appmag(np.sort(data1999tab1['redshift']), mod_appmag), label='Fit Table 1', c='green')
axs[1].set_title('1999 Table 1 Data and Fit')
axs[1].set_xlabel('Redshift')
axs[1].set_ylabel('Apparent magnitude (B-band)')
axs[1].legend()
plt.tight_layout() 
plt.show()

# Question (2) - All Data Combined 
ALL_output_NO_REST = curve_fit(mB_model, All_Redshift, All_appMAG) #, absolute_sigma=True) #, p0 = initial_guess, bounds = bounds,sigma=data1999tab2['appmagcorrerr'], method = 'dogbox') #, absolute_sigma=True)
Omega_M0_T = ALL_output_NO_REST[0][0]
Omega_Lambda0_T = ALL_output_NO_REST[0][1]

print(f" Combined Matter density (Omega_M0): {Omega_M0_T} \nCombined Dark energy density (Omega_Lambda0): {Omega_Lambda0_T}")

# Question (4)

# Generate a dense array of redshift values for a smooth curve
min_redshift = np.min(All_Redshift)
max_redshift = np.max(All_Redshift)
dense_redshifts = np.linspace(min_redshift, max_redshift, 1000)  # 1000 points for smoothness

# Calculate the model predictions using the dense array of redshifts
appMAG_Original = mB_model(dense_redshifts, Omega_M0_T, Omega_Lambda0_T)

cases = [[0.5,0.5],[1,0],[0,1]]
Equal_Case = mB_model(dense_redshifts, cases[0][0], cases[0][1])
All_Matter = mB_model(dense_redshifts, cases[1][0], cases[1][1])
ALL_DEnergy = mB_model(dense_redshifts, cases[2][0], cases[2][1])

# Plot the fit(s) aginst the actual data
fig, ax = plt.subplots()
ax.scatter(All_Redshift, All_appMAG, label='Combined Data', c='blue')
ax.errorbar(All_Redshift, All_appMAG, xerr=All_Redshift_ER, yerr=All_appMAG_ER, fmt='o', capsize=5, label='Error Bars')
ax.plot(dense_redshifts, appMAG_Original, label=r'Fitted Model: $\Omega_M={:.2f}, \Omega_\Lambda={:.2f}$'.format(Omega_M0_T, Omega_Lambda0_T), c='green')
ax.plot(dense_redshifts, Equal_Case, label=r'Fitted Model: $\Omega_M={:.2f}, \Omega_\Lambda={:.2f}$'.format(cases[0][0], cases[0][1]), c='red')
ax.plot(dense_redshifts, All_Matter, label=r'Fitted Model: $\Omega_M={:.2f}, \Omega_\Lambda={:.2f}$'.format(cases[1][0], cases[1][1]), c='black')
ax.plot(dense_redshifts, ALL_DEnergy, label=r'Fitted Model: $\Omega_M={:.2f}, \Omega_\Lambda={:.2f}$'.format(cases[2][0], cases[2][1]), c='yellow')
ax.set_title('Fitted Curve with Combined Data')
ax.set_xlabel('Redshift')
ax.set_ylabel('Apparent magnitude (B-band)')
plt.legend()
plt.show()

# QUESTION (3) - Separate Data Sets 

# Altered DL formula to pass a constraint on the total matter-energy density specifying a flat universe
def forced_mB_model_flat_universe(z, Omega_M0):
    Omega_Lambda0 = 1 - Omega_M0      # Enforcing flat universe constraint
    DL = luminosity_distance(z, Omega_M0, Omega_Lambda0)
    return mod_appmag + 5 * np.log10(DL)

# Intial conditions and constraints to guide the curvefit function
bounds = (0, 1)
initial_guess = [0.3]

output_low = curve_fit(forced_mB_model_flat_universe, data1999tab2['redshift'], data1999tab2['appmagcorr'],p0 = initial_guess, bounds = bounds) #, absolute_sigma=True) #, p0 = initial_guess, bounds = bounds,sigma=data1999tab2['appmagcorrerr'], method = 'dogbox') #, absolute_sigma=True)

Omega_M0_fitted_low = output_low[0]
Omega_Lambda0_fitted_low = 1 - Omega_M0_fitted_low

print(f'Table 2 Low Redshift Flat_UN (matter, dark_EN): {Omega_M0_fitted_low}, {Omega_Lambda0_fitted_low}')

output_high = curve_fit(forced_mB_model_flat_universe, data1999tab1['redshift'], data1999tab1['appmageff'],p0 = initial_guess, bounds = bounds) #, p0 = initial_guess, bounds = bounds,sigma=data1999tab1['appmagefferr']) #, absolute_sigma=True)

Omega_M0_fitted_high = output_high[0]
Omega_Lambda0_fitted_high = 1 - Omega_M0_fitted_high

print(f'Table 1 High Redshift Flat_UN (matter, dark_EN): {Omega_M0_fitted_high}, {Omega_Lambda0_fitted_high}')

# Question (3) - All data combined
bounds = (0, 1)
initial_guess = [0.3]

ALL_output = curve_fit(forced_mB_model_flat_universe, All_Redshift, All_appMAG,p0 = initial_guess, bounds = bounds) #, absolute_sigma=True) #, p0 = initial_guess, bounds = bounds,sigma=data1999tab2['appmagcorrerr'], method = 'dogbox') #, absolute_sigma=True)

Omega_M0_T_FL = ALL_output[0][0]
Omega_Lambda0_T_FL = 1 - Omega_M0_T_FL

print(f" For Flat Universe Model\nCombined Matter density (Omega_M0): {Omega_M0_T_FL} \nCombined Dark energy density (Omega_Lambda0): {Omega_Lambda0_T_FL}")
