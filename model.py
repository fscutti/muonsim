#! /usr/bin/env python3
import matplotlib.pyplot as plot
import numpy as np

from muonsim import muonflux
import sys


def flux_GCCLY(energy, zenith):
    """Guan et al. primary muon flux"""
    c = np.cos(zenith * np.pi / 180)
    return muonflux._flux_gccly(c, energy, 0)


if __name__ == "__main__":
    # check that these parameters are consistent with
    # what has been simulated. Look at the run.py script.
    azimuth = 0
    zenith = np.linspace(0, 90, 101)
    energy = 1  # GeV

    model = np.empty(zenith.size - 1)

    for i, zi in enumerate(zenith[:-1]):
        dz = zenith[i + 1] - zi
        zz = np.linspace(zi, zenith[i + 1], 101)

        ff = np.array([flux_GCCLY(energy + 0.5, zzz) for zzz in zz])

        model[i] = np.trapz(ff, zz) / dz

    zenith = 0.5 * (zenith[1:] + zenith[:-1])

    plot.figure()
    plot.plot(zenith, model, "r-", label="Guan et al.")
    plot.xlabel("Elevation [deg]")
    plot.ylabel("mean flux (GeV$^{-1}$m$^{-2}$sr$^{-1}$s$^{-1}$)")
    plot.legend(loc=1)
    plot.savefig("test-flux.png")

    plot.show()
