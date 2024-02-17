# 4602B_Proj2.py

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from scipy.constants import c

c *= 10**(-3)  # Use units of km/s as in paper

# Low redshift

data1997 = np.genfromtxt('data1997.txt', dtype=[('names', 'U8'),
                                            ('redshift', 'f8'),
                                            ('appmag', 'f8'),
                                            ('appmagerr', 'f8')])

# Note script M subscript B is called the modified apparent magnitude

def appmag(z, mod_appmag):
    '''Equation (2) from assignment'''
    return mod_appmag + 5*np.log10(c*z)

sol = curve_fit(appmag, data1997['redshift'], data1997['appmag'])
mod_appmag = sol[0][0]

fig, ax = plt.subplots()
ax.set_title("Curve fit of Apparent Magnitude")
ax.set_xlabel("Redshift")
ax.set_ylabel("Apparent magnitude (B-band)")
ax.scatter(data1997['redshift'],
           data1997['appmag'],
           label='Perlmutter et. al (1997)',
           c='b')
ax.plot(np.sort(data1997['redshift']),
        appmag(np.sort(data1997['redshift']), mod_appmag),
        label='$m_B(z)$ for $\mathscr{M}_B=%.2f$' % mod_appmag,
        c='g')
plt.legend()
plt.show()

