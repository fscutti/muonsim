import os
import sys
import math

import ROOT
import numpy as np

from copy import copy
from random import uniform
from scipy import stats
from tqdm import tqdm

from muonsim import geometry as geo
from muonsim import utils

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import cmasher

# NOTE: this should be run with the pyrate environment.


def make_mpl_plot(
    hist,
    cmap="cmr.torch",
    savefig=None,
    log_scale=False,
    title="",
    units="",
    z_min=None,
    z_max=None,
    draw_grid=False,
    flip=False,
    colorbar=True,
):
    directory = os.path.dirname(savefig)

    if flip:
        pass
        # hist = copy(diabolical_flip(hist))

    if not os.path.exists(directory):
        os.makedirs(directory)

    n_az_bins = hist.GetXaxis().GetNbins()
    n_el_bins = hist.GetYaxis().GetNbins()

    data = np.empty([n_az_bins, n_el_bins])

    for g_idx, az_idx, el_idx in utils.bin_loop(hist):
        data[az_idx - 1, el_idx - 1] = hist.GetBinContent(g_idx)

    az_min = hist.GetXaxis().GetXmin()
    az_max = hist.GetXaxis().GetXmax()
    az_bin_width = round((az_max - az_min) / n_az_bins)

    el_min = hist.GetYaxis().GetXmin()
    el_max = hist.GetYaxis().GetXmax()
    el_bin_width = round((el_max - el_min) / n_el_bins)

    # Convert angles to suitable format for a polar plot.
    # az = np.arange(az_min, az_max + az_bin_width) * (2 * np.pi / 360)
    # el = np.arange(el_min, el_max + el_bin_width)

    az = np.arange(az_min, az_max + az_bin_width, az_bin_width) * (2 * np.pi / 360)
    el = np.arange(el_min, el_max + el_bin_width, el_bin_width)

    # Determine the tick labels for the elevation angles.
    el_axis = np.abs(el - 90)
    el_axis_spc = np.ceil(abs(el_axis[-1] - el_axis[0]) / 5).astype(int)
    el_label = el_axis[::el_axis_spc]

    # Create an angle meshgrid
    Az, El = np.meshgrid(az, el_axis)

    # Create new figure
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(projection="polar")

    # Set properties of this polar plot to mimic azimuth and elevation
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_thetamin(az_min)
    ax.set_thetamax(az_max)
    ax.set_rgrids(el_label, [r"$%s\degree$" % (abs(90 - x)) for x in el_label])

    cmap = plt.get_cmap(cmap)

    pc = None

    # Create colormesh of flux values
    if log_scale:
        pc = ax.pcolormesh(
            Az,
            El,
            data.T,
            cmap=cmap,
            shading="flat",
            norm=colors.LogNorm(vmin=z_min, vmax=z_max),
        )

    else:
        pc = ax.pcolormesh(
            Az,
            El,
            data.T,
            cmap=cmap,
            shading="flat",
            vmin=z_min,
            vmax=z_max,
        )

    if colorbar:
        cbar = fig.colorbar(pc)
        cbar.set_label(units)

    plt.title(title)

    plt.tight_layout()

    if not draw_grid:
        plt.grid()

    # If savefig is not None, save the figure
    if savefig is not None:
        plt.savefig(savefig, dpi=500)
        plt.close(fig)

    # Else, simply show it
    else:
        plt.show()


def check_edge(h, x_idx, y_idx):
    neighbours = (
        h.GetBinContent(x_idx, y_idx + 1),
        h.GetBinContent(x_idx, y_idx - 1),
        h.GetBinContent(x_idx + 1, y_idx + 1),
        h.GetBinContent(x_idx + 1, y_idx),
        h.GetBinContent(x_idx + 1, y_idx - 1),
        h.GetBinContent(x_idx - 1, y_idx + 1),
        h.GetBinContent(x_idx - 1, y_idx),
        h.GetBinContent(x_idx - 1, y_idx - 1),
    )

    zeroes = 0

    for n in neighbours:
        if n <= 0.0:
            zeroes += 1
            if zeroes == 1:
                break

    return zeroes >= 1


def get_total_flux(h_flux_list):
    h_tot_flux = copy(h_flux_list[0])
    h_tot_flux.Reset()
    h_tot_flux.SetName("h_total_flux")
    h_tot_flux.SetTitle("h_total_flux")

    for glob_idx, x_idx, y_idx in utils.bin_loop(h_tot_flux):
        hm_num, hm_den, hm_unc = 0.0, 0.0, 0.0

        for h_flux in h_flux_list:
            bin_flux = h_flux.GetBinContent(x_idx, y_idx)
            bin_unc = h_flux.GetBinError(x_idx, y_idx)

            # We might want to remove bins at the edges as they cannot
            # represent a meaningful flux.
            bin_is_on_edge = False
            # if self._remove_edge_bins:
            # bin_is_on_edge = check_edge(h_flux, x_idx, y_idx)

            if bin_flux and (not math.isnan(bin_flux)) and (not bin_is_on_edge):
                hm_num += 1.0
                hm_den += 1.0 / bin_flux
                hm_unc += (bin_unc / (bin_flux**2)) ** 2

        if hm_den > 0.0:
            h_tot_flux.SetBinContent(x_idx, y_idx, hm_num / hm_den)

            h_tot_flux.SetBinError(
                x_idx, y_idx, hm_num * math.sqrt(hm_unc) / (hm_den**2)
            )
    return h_tot_flux


# -------------
# Configuration
# -------------
path = "/Users/fscutti/github/muonsim"

input_dir = os.path.join(path, "AllFlux")
output_dir = os.path.join(path, "AllFlux")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_file = ROOT.TFile(os.path.join(input_dir, "Costerfield_fluxes.root"), "READ")

# selection = ["Mod2", "Mod5"]
selection = ["Mod3", "Mod5"]

# ---------------
# Making the plot
# ---------------
h_flux_list = []
for h in input_file.GetListOfKeys():
    # Selecting histograms.
    if any([bool(s in h.GetName()) for s in selection]):
        h_flux_list.append(input_file.Get(h.GetName()))

h_tot_flux = get_total_flux(h_flux_list)

selection_tag = "+".join(selection)

make_mpl_plot(
    h_tot_flux,
    cmap="cmr.ocean",
    savefig=os.path.join(output_dir, h_tot_flux.GetName() + selection_tag),
    log_scale=False,
    draw_grid=True,
    title=f"Total Measured Flux {selection_tag}",
    units="$\mathrm{N_{\mu}/[GeV\;x\;s\;x\;sr\;x\;m^{2}]}$",
)
