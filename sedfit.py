import numpy as np
import pandas as pd
from lmfit import Model, Parameters
from scipy.interpolate import LinearNDInterpolator
from pyneb import RedCorr
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union, Any


def get_catalog_id_and_weff(cols: list) -> Tuple[
    list, List[Union[float, Any]], LinearNDInterpolator, LinearNDInterpolator]:
    """Retrives the waveleneghts, passbands names, and interpolated model fluxes for hot and cool stars."""
    dfwd = pd.read_table("UV_IR_IPHAS__modelmags_WD_logg.dat", sep="\t")
    dfwd.sort_values(["Teff", "logg"], inplace=True)
    dfms = pd.read_table("UV_IR_IPHAS_modelmags_MS_logg.dat", sep="\t")
    dfms.sort_values(["Teff", "logg"], inplace=True)

    parshot = [[dfwd["Teff"][i], dfwd["logg"][i]] for i in range(len(dfwd))]
    mags_hot = [dfwd[cols].T[i] for i in range(len(dfwd[cols]))]
    parscool = [[dfms["Teff"][i], dfms["logg"][i]] for i in range(len(dfms))]
    mags_cool = [dfms[cols].T[i] for i in range(len(dfms[cols]))]

    # Available passbands
    photFilts = ["GALEX_GALEX.FUV", "GALEX_GALEX.NUV",
                 "SLOAN_SDSS.u", "SLOAN_SDSS.g", "SLOAN_SDSS.r", "SLOAN_SDSS.i", "SLOAN_SDSS.z",
                 "PAN-STARRS_PS1.g", "PAN-STARRS_PS1.r", "PAN-STARRS_PS1.i", "PAN-STARRS_PS1.z", "PAN-STARRS_PS1.y",
                 "Generic_Johnson.U", "Generic_Johnson.B", "Generic_Johnson.V", "Generic_Johnson.R",
                 "Generic_Johnson.I",
                 "2MASS_2MASS.J", "2MASS_2MASS.H", "2MASS_2MASS.Ks", "IUE_IUE.1250-1300", "IUE_IUE.1450-1500",
                 "IUE_IUE.1675-1725", "IUE_IUE.2150-2200", "IUE_IUE.2395-2445", "IUE_IUE.2900-2950",
                 "IUE_IUE.2900-3000", "INT_IPHAS.gI", "INT_IPHAS.gR", "INT_IPHAS.Ha",
                 "INT_WFC.RGO_u", "INT_WFC.Gunn_g", "INT_WFC.HeII"]
    # Effective wavelengths that corresponds to the passbands
    waves_eff = [1542.3, 2274.4, 3594.9, 4640.4, 6122.3, 7439.5, 8897.1,
                 4775.6, 6129.5, 7484.6, 8657.8, 9603.1,
                 3570.6, 4378.1, 5466.1, 6695.6, 8565.1,
                 12350.0, 16620.0, 21590.0, 1284.7, 1475.2, 1699.6, 2174.9,
                 2420.2, 2924.8, 2949.9, 7706.80, 6194.03, 6567.98, 3564.87, 4814.24, 4684.89]
    dict_bands_waves = dict(zip(photFilts, waves_eff))

    interpHotMags = LinearNDInterpolator(parshot, mags_hot)
    interpCoolMags = LinearNDInterpolator(parscool, mags_cool)

    return cols, [dict_bands_waves[b] for b in cols], interpHotMags, interpCoolMags


def unred(wave: np.ndarray, flux: np.ndarray, ebv: float, law: str = "CCM89", flux_type: str = "flux"):
    """Unredening the flux/magnitude."""
    rc = RedCorr(E_BV=ebv, law=law)
    fact = rc.getCorr(wave)

    if flux_type == "flux":
        return flux * fact
    elif flux_type == "ABmag":
        cAA = 2.99792458e+18
        flux = 10 ** (-0.4 * (48.6 + flux)) * cAA / wave ** 2
        flux = flux * fact
        return -2.5 * np.log10(flux * wave ** 2 / cAA) - 48.6


def estimate_ML(ppd: List[float]) -> dict:
    """Calcualtes the Maximum-Likelihood of a posterior probability data."""
    result_dict = {}

    highest_prob = np.argmax(ppd.lnprob)
    hp_loc = np.unravel_index(highest_prob, ppd.lnprob.shape)
    mle_soln = ppd.chain[hp_loc]
    pars = [p for p in ppd.params if ppd.params[p].vary]
    print("\nMaximum Likelihood Estimation (MLE):")
    print("----------------------------------")
    for ix, param in enumerate(pars):
        # quantiles
        quantiles = np.percentile(ppd.flatchain[param], [2.28, 15.9, 50, 84.2, 97.7])
        one_sigma_error = 0.5 * (quantiles[3] - quantiles[1])

        print(f"{param}: {mle_soln[ix]:.3f}+/-{one_sigma_error}")

        result_dict[param] = [mle_soln[ix], one_sigma_error]

    return result_dict


class ModelFit(ABC):
    @abstractmethod
    def __init__(self, passbands: List[str]):
        self.passbands = passbands

    @abstractmethod
    def initial_parameters(self, *args):
        pass

    @abstractmethod
    def model(self, *args):
        pass


class ModelIndexes(ModelFit):
    def __init__(self, passbands):
        self.passbands = passbands
        _, self.waves_eff, self.interpHotFlux, self.interpCoolFlux = get_catalog_id_and_weff(self.passbands)

    def initial_parameters(self, thot: float, tcool: float, beta: float, ebv: float, vary_ebv: bool,
                           is_binary: bool, thot_p: List[float], tcool_p: List[float], beta_p: List[float],
                           ebv_p: List[float]) -> dict:
        """Create the parameters for the model. Returns a Parameters lmfit dictionary."""
        pars = Parameters()
        # Parameters -> Name, Value, Vary, Min, Max
        pars.add_many(
            ("thot", thot, True, thot_p[0], thot_p[1]),
            ("tcool", tcool, is_binary, tcool_p[0], tcool_p[1]),
            ("beta", beta if is_binary else 0., is_binary, beta_p[0], beta_p[1]),
            ("ebv", ebv, vary_ebv, ebv_p[0], ebv_p[1])
        )

        return pars

    def model(self, x: List[float], thot: float, tcool: float, beta: float, ebv: float):
        # By setting the negative value of ebv we are actually reddened the flux models.
        flux_P1 = unred(self.waves_eff, self.interpHotFlux(thot, 7.0), -ebv)
        flux_S1 = unred(self.waves_eff, self.interpCoolFlux(tcool, 5.0), -ebv)

        flux_ratios = []
        for i in range(len(self.passbands)):
            for k in range(i, len(self.passbands) - 1):
                num = flux_P1[i] + flux_S1[i] * beta ** 2.0
                den = flux_P1[k + 1] + flux_S1[k + 1] * beta ** 2.0
                flux_ratios.append(num / den)
        return -2.5 * np.log10(np.asarray(flux_ratios))


class ModelPogson(ModelFit):
    def __init__(self, passbands):
        self.passbands = passbands
        _, self.waves_eff, self.interpHotFlux, self.interpCoolFlux = get_catalog_id_and_weff(self.passbands)

    def initial_parameters(self, thot: float, tcool: float, beta: float, ebv: float, vary_ebv: bool,
                           is_binary: bool, thot_p: List[float], tcool_p: List[float], beta_p: List[float],
                           ebv_p: List[float], alpha: float, alpha_p: List[float], vary_alpha: bool) -> dict:
        """Create the parameters for the model. Returns a Parameters lmfit dictionary."""
        pars = Parameters()
        # Parameters -> Name, Value, Vary, Min, Max
        pars.add_many(
            ("thot", thot, True, thot_p[0], thot_p[1]),
            ("tcool", tcool, is_binary, tcool_p[0], tcool_p[1]),
            ("beta", beta if is_binary else 0., is_binary, beta_p[0], beta_p[1]),
            ("alpha", vary_alpha, True, alpha_p[0], alpha_p[1]),
            ("ebv", ebv, vary_ebv, ebv_p[0], ebv_p[1])
        )

        return pars

    def model(self, x, thot: float, tcool: float, beta: float, ebv: float, alpha: float):
        """Only works for ABmag system."""
        flux_hot = unred(self.waves_eff, self.interpHotFlux(thot, 7.0), -ebv)
        flux_cool = unred(self.waves_eff, self.interpCoolFlux(tcool, 5.0), -ebv)
        return -2.5 * np.log10((flux_hot + flux_cool * beta ** 2) * np.exp(alpha) ** 2) - 48.6


class FitSED:
    def __init__(self, mags: np.ndarray, err_mags: np.ndarray, passbands: List[str], pn_name: str = "Object 1",
                 kernel_model: str = "indexes"):
        self.mags = mags
        self.err_mags = err_mags
        self.passbands = passbands
        self.pn_name = pn_name
        self.kernel_model = kernel_model

    def prepare_run(self, thot: float = 50000., alpha: float = -36, tcool: float = 2600, beta: float = 0,
                    ebv: float = 0., vary_ebv: bool = True, is_binary: bool = False,
                    thot_p: List[float] = (20000., 200000.), tcool_p: List[float] = (2600., 10000.),
                    beta_p: List[float] = (0.0, np.inf), alpha_p: List[float] = (-np.inf, np.inf),
                    vary_alpha: bool = True, ebv_p: List[float] = (0.0, 1.0)):
        pars, mod = None, None
        if self.kernel_model == "indexes":
            kernel = ModelIndexes(self.passbands)
            pars = kernel.initial_parameters(thot=thot, tcool=tcool, beta=beta, ebv=ebv, is_binary=is_binary,
                                             vary_ebv=vary_ebv, thot_p=thot_p, tcool_p=tcool_p, beta_p=beta_p,
                                             ebv_p=ebv_p)
            mod = kernel.model
        elif self.kernel_model == "pogson":
            kernel = ModelPogson(self.passbands)
            pars = kernel.initial_parameters(thot=thot, tcool=tcool, beta=beta, ebv=ebv, is_binary=is_binary,
                                             vary_ebv=vary_ebv, thot_p=thot_p, tcool_p=tcool_p, beta_p=beta_p,
                                             ebv_p=ebv_p, alpha=alpha, alpha_p=alpha_p, vary_alpha=vary_alpha)
            mod = kernel.model

        return pars, mod

    def __prepare_data(self):
        if self.kernel_model == "indexes":
            # if not vary_ebv:
            #     mags = unred_mag(waves_eff, mags, Ebv=Ebv)
            # Comparing colors
            mdiffs = []
            mdiffs_err = []
            tmpmag1 = self.mags[0:-1]
            tmpmag2 = self.mags[1:]
            tmpmerr1 = self.err_mags[0:-1]
            tmpmerr2 = self.err_mags[1:]
            for i in range(len(tmpmag1)):
                for k in range(i, len(tmpmag2)):
                    mdiffs.append(tmpmag1[i] - tmpmag2[k])
                    mdiffs_err.append(np.sqrt(tmpmerr1[i] ** 2.0 + tmpmerr2[k] ** 2.0))
            y = np.asarray(mdiffs)
            y_err = np.asarray(mdiffs_err)

        elif self.kernel_model == "pogson":
            y = self.mags
            y_err = self.err_mags
        else:
            y, y_err = None, None

        return y, y_err

    def run_fit(self, pars: dict, model: Callable[[List[str]], List[float]],
                emcee_kws=None, use_weights=False):
        if emcee_kws is None:
            emcee_kws = dict(steps=5000, burn=500, thin=20, is_weighted=False, progress=True, workers=10)
        emcee_params = pars

        print("Preparing input data.")
        y, y_err = self.__prepare_data()
        x = np.array([i for i in range(len(y))])

        weights = None
        if use_weights:
            weights = 1 / y_err

        print(f"Maximizing the kernel_model: {self.kernel_model}")
        mod = Model(model)
        mod_max = mod.fit(data=y, params=emcee_params.copy(), x=x, method="Nelder", nan_policy="omit",
                          weights=weights)
        print(mod_max.fit_report())

        print("Running MCMC...")
        # emcee_params = mod_max.params.copy()

        # Noise prior defined according to user input errors.
        emcee_kws["is_weighted"] = True
        if weights is None:
            emcee_kws["is_weighted"] = False
            emcee_params.add("__lnsigma", value=np.log(np.mean(self.err_mags)), min=np.log(min(self.err_mags)),
                             max=np.log(max(self.err_mags)))

        # running MCMC algorithm.
        results_emcee = mod.fit(data=y, x=x, params=emcee_params, method='emcee', nan_policy='omit',
                                fit_kws=emcee_kws, weights=weights)
        print(results_emcee.fit_report())

        return results_emcee

    #TODO: Implement the plot function in this method.

    # def plot_fit(self, chain:np.ndarray, model: Callable[[float, float], np.ndarray], pars: dict = None) -> None:
    #     """Plotting best fit result."""
    #     if pars is not None:
    #         for parameter in pars.keys():
    #             if parameter in chain.params:
    #                 chain.params[parameter].value = pars[parameter][0]
    #                 chain.params[parameter].stdderr = pars[parameter][1]
    #     pass