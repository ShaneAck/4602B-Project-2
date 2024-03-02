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
# Combined Data (Table 1 & 2 ) for later parts 
All_Redshift = np.concatenate((data1999tab1['redshift'], data1999tab2['redshift']))
All_Redshift_ER = np.concatenate((data1999tab1['redshifterr'], data1999tab2['redshifterr']))
All_appMAG = np.concatenate((data1999tab1['appmageff'], data1999tab2['appmagcorr']))
All_appMAG_ER = np.concatenate((data1999tab1['appmagefferr'], data1999tab2['appmagcorrerr']))

# Note script M subscript B is called the modified apparent magnitude

# QUESTION (1): Shane 

def appmag(z, mod_appmag):
    '''Equation (2) from assignment'''
    return mod_appmag + 5*np.log10(c*z)

sol = curve_fit(appmag,
                data1999tab2['redshift'],
                data1999tab2['appmagcorr'],
                sigma=data1999tab2['appmagcorrerr'],
                absolute_sigma=True)
mod_appmag = sol[0][0]
fig, ax = plt.subplots()
ax.set_title("Curve fit of Apparent Magnitude")
ax.set_xlabel("Redshift")
ax.set_ylabel("Apparent magnitude (B-band)")
ax.scatter(data1999tab2['redshift'],
           data1999tab2['appmagcorr'],
           label='Perlmutter et. al (1999)',
           c='b',
           zorder=1)
ax.plot(np.sort(data1999tab2['redshift']),
        appmag(np.sort(data1999tab2['redshift']), mod_appmag),
        label='$m_B(z)$ for $\mathscr{M}_B=%.2f$' % mod_appmag,
        c='g',
        zorder=0)
plt.legend()
plt.savefig('4602B_Proj2_Fig1.png', dpi=150)
plt.show()

# QUESTION (2): Dakota and Shane

# Function to define the integrand in the DL (lumminosity distance) formula
def integrand(z_prime, Omega_M0, Omega_Lambda0):
    return 1/np.sqrt((Omega_M0) * ((1 + z_prime)**3) + Omega_Lambda0)

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
            integral, _ = quad(integrand, 0, z_val, args=(Omega_M0, Omega_Lambda0))
            DL_array[i] = ((1 + z_val) * c ) * (integral)
        return DL_array

# Apparent magnitude model mB(z) using equation (1)
def mB_model(z, Omega_M0, Omega_Lambda0):
    D_L = luminosity_distance(z, Omega_M0, Omega_Lambda0)
    return mod_appmag + 5 * np.log10(D_L)

ALL_output_NO_REST = curve_fit(mB_model, All_Redshift, All_appMAG)
Omega_M0_T = ALL_output_NO_REST[0][0]
Omega_Lambda0_T = ALL_output_NO_REST[0][1]

print(" Combined matter density (Omega_M0)           = %.4f" % Omega_M0_T)
print(" Combined dark energy density (Omega_Lambda0) = %.4f" % Omega_Lambda0_T)

# QUESTION (4) - Dakota

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
ax.scatter(All_Redshift, All_appMAG,
           label='Combined Data',
           c='blue', linewidth=0.5, zorder=5)
ax.errorbar(All_Redshift, All_appMAG,
            xerr=All_Redshift_ER, yerr=All_appMAG_ER,
            fmt='none', linewidth=1, capsize=3, zorder=4, color='blue')
ax.plot(dense_redshifts, appMAG_Original,
        label='$(\Omega_{M,0}, \Omega_{\Lambda,0})=(%.2f, %.2f)$' % (Omega_M0_T, Omega_Lambda0_T),
        c='black', zorder=3)
ax.plot(dense_redshifts, Equal_Case,
        label='$(\Omega_{M,0}, \Omega_{\Lambda,0})=(%.2f, %.2f)$' % (cases[0][0], cases[0][1]),
        c='#ff42c0', zorder=1)
ax.plot(dense_redshifts, All_Matter,
        label='$(\Omega_{M,0}, \Omega_{\Lambda,0})=(%.2f, %.2f)$' % (cases[1][0], cases[1][1]),
        c='#428eff', zorder=2)
ax.plot(dense_redshifts, ALL_DEnergy,
        label='$(\Omega_{M,0}, \Omega_{\Lambda,0})=(%.2f, %.2f)$' % (cases[2][0], cases[2][1]),
        c='#ff9a42', zorder=0)
ax.set_title('Fitted Curve with Combined Data')
ax.set_xlabel('Redshift')
ax.set_ylabel('Apparent magnitude (B-band)')
plt.legend()
plt.savefig('4602B_Proj2_Fig2.png', dpi=150)
plt.show()

# QUESTION (3) - Dakota

# Altered DL formula to pass a constraint on the total matter-energy density specifying a flat universe
def forced_mB_model_flat_universe(z, Omega_M0):
    Omega_Lambda0 = 1 - Omega_M0      # Enforcing flat universe constraint
    DL = luminosity_distance(z, Omega_M0, Omega_Lambda0)
    return mod_appmag + 5 * np.log10(DL)

bounds = (0, 1)
initial_guess = [0.3]

ALL_output = curve_fit(forced_mB_model_flat_universe,
                       All_Redshift, All_appMAG,
                       p0 = initial_guess,
                       bounds = bounds)

Omega_M0_T_FL = ALL_output[0][0]
Omega_Lambda0_T_FL = 1 - Omega_M0_T_FL

print("For flat universe model:")
print(" Combined matter density (Omega_M0)           = %.4f" % Omega_M0_T_FL)
print(" Combined dark energy density (Omega_Lambda0) = %.4f" % Omega_Lambda0_T_FL)
