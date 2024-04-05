import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as pt

img_dim = 79 # fcg size after preproc
R = img_dim // 2 # max radius

# max kernel size
K = 20 # rho_0 from preproc

# area under exponential drop-off curve
# areas should equal integral of K*np.exp(-decay_rate*r) from r = 0 to R
areas = np.linspace(0, R*K, 11)[1:-1]

# solve for decay rates that realize given areas, based on examples in
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lambertw.html
a = K / areas
b = -a
c = -R
decay_rates = np.real(a - lambertw(-b*c*np.exp(a*c))/c)

# decay_rate = -np.log((min_ks+1) / (max_ks+1)) / rad
# powers = np.array([.01, .05, .1, .5, 1, 5, 10, 100])

r = np.arange(0,R+1,0.1)
pt.plot(r, [K]*len(r), 'k:', label='full con')
for decay_rate in decay_rates:
    fr = K * np.exp(-decay_rate * r)
    portion = fr.mean() / K
    # pt.plot(r, fr, '-', label=f"{np.real(decay_rate):e}")
    pt.plot(r, fr, '-', label=f"{portion:.3f}")
# pt.plot(np.arange(rad+1), max_ks * np.exp(-decay_rate * np.arange(rad+1)), 'r-')
# for power in powers:
#     pt.plot(np.arange(rad+1), max_ks * np.exp(-power * decay_rate * np.arange(rad+1)), '-', label=str(power))
pt.legend()
pt.show()

