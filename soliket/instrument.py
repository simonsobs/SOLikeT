import numpy as np
from scipy import constants
from cobaya.theory import Theory


def _cmb2bb(nu):
    # NB: numerical factors not included
    T_CMB = 2.72548
    x = nu * constants.h * 1e9 / constants.k / T_CMB
    return np.exp(x) * (nu * x / np.expm1(x)) ** 2


class Instrument(Theory):
    def _calculate_bandint_freqs(self, **kwargs):
        raise NotImplementedError

    def get_can_provide(self):
        return ["bandint_freqs"]

    # def get_bandint_freqs(self):
    #     return self._calculate_bandint_freqs(**params_values_dict)

    def calculate(self, state, want_derived=False, **params_values_dict):
        # calculate bandint_freq here, based on params
        state["bandint_freqs"] = self._calculate_bandint_freqs(**params_values_dict)


class LAT(Instrument):
    bandint_width = 0.3
    bandint_nstep = 100

    freqs = [93, 145, 225]
    params = dict(bandint_shift_93=0.0, bandint_shift_145=0.0, bandint_shift_225=0.0)

    def _calculate_bandint_freqs(self, **params_values):
        # Bandpass construction
        if not hasattr(self.bandint_width, "__len__"):
            self.bandint_width = np.full_like(np.array(self.freqs), self.bandint_width, dtype=np.float)
        if np.any(np.array(self.bandint_width) > 0):
            assert self.bandint_nstep > 1, "bandint_width and bandint_nstep not coherent"
            assert np.all(
                np.array(self.bandint_width) > 0
            ), "one band has width = 0, set a positive width and run again"

            self.bandint_freqs = []
            for ifr, fr in enumerate(self.freqs):
                bandpar = "bandint_shift_" + str(fr)
                bandlow = fr * (1 - self.bandint_width[ifr] * 0.5)
                bandhigh = fr * (1 + self.bandint_width[ifr] * 0.5)
                print(bandlow, bandhigh)
                nubtrue = np.linspace(bandlow, bandhigh, self.bandint_nstep)
                nub = np.linspace(
                    bandlow + params_values[bandpar], bandhigh + params_values[bandpar], self.bandint_nstep
                )
                tranb = _cmb2bb(nub)
                tranb_norm = np.trapz(_cmb2bb(nubtrue), nubtrue)
                self.bandint_freqs.append([nub, tranb / tranb_norm])
        else:
            self.bandint_freqs = np.empty_like(self.freqs)
            for ifr, fr in enumerate(self.freqs):
                bandpar = "bandint_shift_" + str(fr)
                self.bandint_freqs[ifr] = fr + params_values[bandpar]

