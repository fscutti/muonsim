"""This module contains all functions needed to define the muon flux.
The formula is taken from https://arxiv.org/abs/1509.06176v1."""
import numpy as np
import math


def _charge_fraction(charge):
    """This function calculates the fraction of the muon flux for a given charge."""
    charge_ratio = 1.2766

    if charge < 0:
        return 1 / (1 + charge_ratio)

    elif charge > 0:
        return charge_ratio / (1 + charge_ratio)

    else:  # pragma: no cover
        return 1


def flux_gaisser(cos_theta, kinetic_energy, charge):
    """This function provides the Gaisser's flux model."""
    Emu = kinetic_energy + 0.10566
    ec = 1.1 * Emu * cos_theta
    rpi = 1 + ec / 115
    rK = 1 + ec / 850
    return 1.4e03 * pow(Emu, -2.7) * (1 / rpi + 0.054 / rK) * _charge_fraction(charge)


def _cos_theta_star(cos_theta):
    """This function calculates Volkova's parametrization of the cosine of theta."""
    p = [0.102573, -0.068287, 0.958633, 0.0407253, 0.817285]
    cs2 = (
        cos_theta * cos_theta
        + p[0] * p[0]
        + p[1] * pow(cos_theta, p[2])
        + p[3] * pow(cos_theta, p[4])
    ) / (1 + p[0] * p[0] + p[1] + p[3])
    return np.sqrt(cs2) if cs2 > 0 else 0


def _flux_gccly(cos_theta, kinetic_energy, charge):
    """This function provides the Guan et al. parametrization of the sea level flux."""
    # Since this function might be needed for MC generation, we will need to
    # discard unphysical values by hand.

    ## Added. The flux is zero for unphysical parameters.
    if kinetic_energy < 0 or abs(cos_theta) > 1 or cos_theta < 0:
        return 0.0

    Emu = kinetic_energy + 0.10566
    cs = _cos_theta_star(cos_theta)

    ## Previsouly added.
    # if cs == 0:
    #    return 0

    return pow(1 + 3.64 / (Emu * pow(cs, 1.29)), -2.7) * flux_gaisser(
        cs, kinetic_energy, charge
    )


def sea_level(sample):
    """Muon flux at sea level."""
    charge = 1 if np.random.uniform(0, 1) > 0.5 else -1
    # return _flux_gccly(sample, 10e-4,  charge)
    return _flux_gccly(*sample, charge)


def flat(sample):
    """Flat flux."""
    cos_theta, energy = sample

    if abs(cos_theta) > 1 or cos_theta < 0:
        return 0.0

    if energy < 0 or energy > 100:
        return 0.0

    return 1.0
