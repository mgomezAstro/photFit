## photFit

Photometry fitting python pckage. This package is intended to fit the spectral energy distribution of any one- or two-body system using a theoretial atmopshere models, for a plenty of photmetric bands, of hot and cool stars (e.g., TLUSTY, NEXTGEN, etc...). It also includes a visualization tool to inspect the best fitting parameters (TODO).

### Usage

This is a ```main.py``` example.
``` python
import numpy as np
from sedfit import FitSED, estimate_ML

mags = np.asarray([19.854, 21.4608, 20.18, 20.27, 20.58, 20.26])
merr = np.asarray([0.157, 0.2887, 0.03, 0.05, 0.12, 0.09])

fs = FitSED(mags=mags,
            err_mags=merr,
            passbands=["GALEX_GALEX.FUV",
                       "GALEX_GALEX.NUV",
                       "INT_WFC.Gunn_g",
                       "INT_IPHAS.gR",
                       "INT_IPHAS.Ha",
                       "INT_IPHAS.gI"],
            kernel_model="indexes",
            pn_name=object_name))

# Initial values
pars, mod = fs.prepare_run(thot=90000., tcool=7500., is_binary=True, vary_ebv=True, ebv=0.2, ebv_p=[0.0, 0.62], beta=10.)

res1 = fs.run_fit(pars, mod, emcee_kws=dict(steps=7000, burn=500, nwalkers=100, progress=True, workers=5), use_weights=True)

best_res = estimate_ML(res1)
```

This will print the best fitting values as well as the ML estimation.

### Available kernels

- Indexes
<!-- $$
m_{\rm A}-m_{\rm B} = -2.5\log\left( \frac{f^{1}_{\rm A} + f^{2}_{\rm A}\beta^2}{f^{1}_{\rm B} + f^{2}_{\rm B}\beta^2} \right),
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=m_%7B%5Crm%20A%7D-m_%7B%5Crm%20B%7D%20%3D%20-2.5%5Clog%5Cleft(%20%5Cfrac%7Bf%5E%7B1%7D_%7B%5Crm%20A%7D%20%2B%20f%5E%7B2%7D_%7B%5Crm%20A%7D%5Cbeta%5E2%7D%7Bf%5E%7B1%7D_%7B%5Crm%20B%7D%20%2B%20f%5E%7B2%7D_%7B%5Crm%20B%7D%5Cbeta%5E2%7D%20%5Cright)%2C"></div>

where <!-- $\beta\equiv R_{2}/R_{1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbeta%5Cequiv%20R_%7B2%7D%2FR_%7B1%7D">. This definition is for a two-body SED in the colour A-B filters. In case of one star <!-- $\beta = 0$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbeta%20%3D%200">.


- Pogson
<!-- $$ 
m_{\rm A} = -2.5\log\left( \left[f^{1}_{A} + f^{2}_{\rm A}\beta^2 \right]\alpha^2\right),
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=m_%7B%5Crm%20A%7D%20%3D%20-2.5%5Clog%5Cleft(%20%5Cleft%5Bf%5E%7B1%7D_%7BA%7D%20%2B%20f%5E%7B2%7D_%7B%5Crm%20A%7D%5Cbeta%5E2%20%5Cright%5D%5Calpha%5E2%5Cright)%2C"></div>

where <!-- $\alpha\equiv R_{1}/D$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Calpha%5Cequiv%20R_%7B1%7D%2FD">, with D the distance to the star.

### To do list

- [x] Creating a githug repository.
- [x] First release.
- [ ] Adding a visualization tool.
- [ ] Refactoring.
- [ ] Public the first release.