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

regions = geo.barmod_telescope_v2.regions

path = "/Users/fscutti/github/muonsim/data"

file_name = "Mod2_119.log"
acceptance_file = "AltCosterfieldEffMap_5000000_5x5_Strip.root"


def find_tag(line, delimiter="   "):
    sensor_tag = ["A", "B", "C", "D", "E"]
    line = line.replace(" ", "")
    return sensor_tag[line.index("X")]


def get_hit(x_region, y_region, z_coord):
    x_hit = uniform(*x_region)
    y_hit = uniform(*y_region)
    z_hit = z_coord
    return np.array([x_hit, y_hit, z_hit])


def get_zenith(x1, y1, z1, x2, y2, z2):
    x = x1 - x2
    y = y1 - y2
    z = z1 - z2
    return 180.0 * math.acos(z / math.sqrt(x**2 + y**2 + z**2)) / np.pi


def get_azimuth(x1, y1, z1, x2, y2, z2):
    x = x1 - x2
    y = y1 - y2
    z = z1 - z2
    phi = 180.0 * math.atan2(x, y) / np.pi

    if phi < 0.0:
        phi = 360.0 + phi

    return phi


h_name = file_name.split(".")[0]

h_counts = ROOT.TH2F(f"h_flux_{h_name}", f"h_flux_{h_name}", 360, 0, 360, 90, 0, 90)

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
        #hist = copy(diabolical_flip(hist))

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
       



def get_acceptance_counts_hist(counts_hist, file_path):
    """The total number of entries in the returned histogram is the number
    of muons one expects to see."""

    acc_counts_hist = copy(counts_hist)
    acc_counts_hist.Reset()
    acc_counts_hist.SetNameTitle("h_acc_counts", "h_acc_counts")

    acceptance_file = ROOT.TFile.Open(file_path, "READ")
    acceptance_hist = acceptance_file.Get("p2_ang_acc_vs_true")

    for g_idx, az_idx, el_idx in utils.bin_loop(counts_hist):
        # In the flux histogram el=y, az=x.
        az = counts_hist.GetXaxis().GetBinCenter(az_idx)
        el = counts_hist.GetYaxis().GetBinCenter(el_idx)

        # In the acceptance histogram el=x, az=y.
        acc_g_idx = acceptance_hist.FindBin(90.0 - el, az)
        acceptance = acceptance_hist.GetBinContent(acc_g_idx)

        n_muons = counts_hist.GetBinContent(g_idx)
        n_acc_muons = n_muons * acceptance
        acc_counts_hist.SetBinContent(g_idx, n_acc_muons)
        acc_counts_hist.SetBinError(g_idx, math.sqrt(n_acc_muons))

    return acc_counts_hist


def get_flux_hist(count_hist, time=1, surface=0.15 * 0.15):
    flux_hist = copy(count_hist)
    flux_hist.Reset()
    flux_hist.SetNameTitle("h_flux", "h_flux")

    # Normalisation constants.
    deg = np.pi / 180.0
    energy_factor = pow(10, 4) - pow(10, -3)

    surface_factor = surface
    time_factor = time

    for g_idx, az_idx, el_idx in utils.bin_loop(count_hist):
        # In the flux histogram el=y, az=x.
        az = flux_hist.GetXaxis().GetBinCenter(az_idx)
        el = flux_hist.GetYaxis().GetBinCenter(el_idx)

        az_up = flux_hist.GetXaxis().GetBinUpEdge(az_idx)
        az_low = flux_hist.GetXaxis().GetBinLowEdge(az_idx)

        el_up = flux_hist.GetYaxis().GetBinUpEdge(el_idx)
        el_low = flux_hist.GetYaxis().GetBinLowEdge(el_idx)

        n_muons = count_hist.GetBinContent(g_idx)

        norm = energy_factor
        norm *= surface_factor
        norm *= time_factor
        norm *= (
            deg * np.fabs(np.sin(el_up * deg) - np.sin(el_low * deg)) * (az_up - az_low)
        )

        flux_hist.SetBinContent(g_idx, n_muons / norm)
        flux_hist.SetBinError(g_idx, math.sqrt(n_muons) / norm)

    return flux_hist


with open(os.path.join(path, file_name)) as file:
    for line in file:
        if "...." in line:
            upper_panel, lower_panel = line.split("....")

            upper_panel_top_layer, upper_panel_lower_layer = upper_panel.split("|")
            lower_panel_top_layer, lower_panel_lower_layer = lower_panel.split("|")

            uptl = find_tag(upper_panel_top_layer)
            upll = find_tag(upper_panel_lower_layer)
            lptl = find_tag(lower_panel_top_layer)
            lpll = find_tag(lower_panel_lower_layer)

            region_name = f"Top_T{uptl}_Top_B{upll}_Bottom_T{lptl}_Bottom_B{lpll}"

            top_region, bottom_region = regions[region_name]

            top_hit = get_hit(*top_region)
            bottom_hit = get_hit(*bottom_region)

            muon = top_hit - bottom_hit

            muon_elevation = 90.0 - get_zenith(*top_hit, *bottom_hit)
            muon_azimuth = get_azimuth(*top_hit, *bottom_hit)

            h_counts.Fill(muon_azimuth, muon_elevation)

h_acc_counts = get_acceptance_counts_hist(h_counts, os.path.join(path, acceptance_file))
h_flux = get_flux_hist(h_acc_counts)

h_flux.Print()

plot_dir = "TestFlux"

make_mpl_plot(
        h_flux,
        savefig=os.path.join(plot_dir, h_flux.GetName()),
        log_scale=False,
        title="Muon Flux",
        units="$\mathrm{N_{\mu}/[GeV\;x\;s\;x\;sr\;x\;m^{2}]}$",
    )


















