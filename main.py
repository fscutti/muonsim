from muonsim import mcmc
from muonsim import muonflux
from muonsim import geometry as geo
from muonsim import histograms as hist
from muonsim import plotter

from muonsim.Detector import Detector

import os
import math
import sys
import matplotlib.pyplot as plt
import numpy as np

import ROOT as R

from tqdm import tqdm

# -------------------------------
# Setting up Metropolis-Hastings
# -------------------------------
np.random.seed(42)

# Initial guess for the sample and number of MCMC samples
initial_sample_guess = np.array([0.9, 100])
proposal_std = [0.01, 10]
num_samples = 1_000_000
#num_samples = 10000
burning = int(num_samples * 0.01)
flux_model = muonflux.sea_level

# -------------------------------
# Setting up muon loop
# -------------------------------
# detector = Detector(geo.block_telescope.detector)
detector = Detector(geo.strip_telescope.detector, geo.strip_telescope.connections)

# Maximum amount of muons in memory.
clear_muons = 1000
# Require coincidence of these specific modules.
# event_modules = ["T12", "B12"]
event_modules = []
coincidences = [2]

# -------------------------------
# Setting up plotting
# -------------------------------
file_name = f"MuonSimTest_{num_samples}_{geo.block_telescope.n_sensors}x{geo.block_telescope.n_sensors}_all_sensors.root"
output_directory = file_name.split(".")[0]


def muon_loop(muons, detector, clear_muons=1000):
    """Loop over generated muons."""
    for m_idx, m in tqdm(enumerate(muons), total=len(muons), colour="red"):
        cos_theta, energy = m

        if m_idx % clear_muons == 0:
            detector.clear_muons(clear_muons)

        # Geometrical properties of the true muon.
        # ----------------------------------------
        # Here units are in degrees.
        muon_true_theta = 180.0 * math.acos(cos_theta) / np.pi
        muon_true_phi = np.random.uniform(0.0, 360.0)

        # Generating the muon on the top panel.
        muon_true_origin = np.array(
            [
                np.random.uniform(*detector.volume["x"]),
                np.random.uniform(*detector.volume["y"]),
                max(detector.volume["z"]),
            ]
        )

        # Clearing all event-related data structures.
        detector.reset_event()

        # This is loading the muon event into memory.
        event_points = detector.intersect(
            muon_true_theta,
            muon_true_phi,
            muon_true_origin,
            coincidences=coincidences,
            event_modules=event_modules,
        )

        event_points = detector.get_element_intersections()
        event_reconstruction = detector.get_event_reconstruction()

        total_path_length = 0
        top_path_length = 0
        bottom_path_length = 0

        hist.true_theta.Fill(muon_true_theta)
        hist.cos_theta.Fill(cos_theta)
        hist.energy.Fill(energy)

        # if the event satisfies all our selection criteria.
        if event_points and event_reconstruction:
            muon_reco_theta, muon_reco_phi = event_reconstruction

            # Filling histograms for individual elements.
            for element, points in event_points.items():
                start, stop = points
                path = np.linalg.norm(start - stop)

                total_path_length += path

                if element.startswith("T"):
                    top_path_length += path

                elif element.startswith("B"):
                    bottom_path_length += path

                hist.path_length(detector, element).Fill(path)

            phi_rel_resolution = 0
            if muon_true_phi > 0:
                phi_rel_resolution = abs(
                    (muon_true_phi - muon_reco_phi) / muon_true_phi
                )
            phi_resolution = muon_true_phi - muon_reco_phi

            theta_rel_resolution = 0
            if muon_true_theta > 0:
                theta_rel_resolution = abs(
                    (muon_true_theta - muon_reco_theta) / muon_true_theta
                )
            theta_resolution = muon_true_theta - muon_reco_theta

            # Filling 1D histograms for all generated muons.
            #hist.energy.Fill(energy)
            #hist.cos_theta.Fill(cos_theta)
            #hist.true_theta.Fill(muon_true_theta)
            hist.reco_theta.Fill(muon_reco_theta)

            hist.total_path_length.Fill(total_path_length)
            hist.top_path_length.Fill(top_path_length)
            hist.bottom_path_length.Fill(bottom_path_length)

            # Filling 2D histograms for all generated muons.
            hist.true_vs_reco_theta.Fill(muon_true_theta, muon_reco_theta)
            hist.true_vs_reco_phi.Fill(muon_true_phi, muon_reco_phi)

            hist.prof_true_vs_reco_theta.Fill(muon_true_theta, muon_reco_theta)
            hist.prof_true_vs_reco_phi.Fill(muon_true_phi, muon_reco_phi)

            hist.prof_theta_resolution.Fill(muon_true_theta, theta_resolution)
            hist.prof_phi_resolution.Fill(muon_true_phi, phi_resolution)

            hist.prof_theta_rel_resolution.Fill(muon_true_theta, theta_rel_resolution)
            hist.prof_phi_rel_resolution.Fill(muon_true_phi, phi_rel_resolution)


def make_plots(detector, file_name, output_directory=None):
    """Plotting. It includes displaying the detector and muon rays."""

    # detector.plot(
    #    add_elements=True, add_muons=True, add_intersections=True, add_connections=False
    # )

    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    else:
        output_directory = "./"

    file_name = os.path.join(output_directory, file_name)

    out_file = R.TFile.Open(file_name, "RECREATE")

    out_file.WriteObject(hist.energy, hist.energy.GetName())
    out_file.WriteObject(hist.cos_theta, hist.cos_theta.GetName())

    out_file.WriteObject(hist.true_theta, hist.true_theta.GetName())
    out_file.WriteObject(hist.reco_theta, hist.reco_theta.GetName())

    out_file.WriteObject(hist.total_path_length, hist.total_path_length.GetName())
    out_file.WriteObject(hist.top_path_length, hist.top_path_length.GetName())
    out_file.WriteObject(hist.bottom_path_length, hist.bottom_path_length.GetName())

    out_file.WriteObject(hist.true_vs_reco_theta, hist.true_vs_reco_theta.GetName())
    out_file.WriteObject(hist.true_vs_reco_phi, hist.true_vs_reco_phi.GetName())

    out_file.WriteObject(
        hist.prof_true_vs_reco_theta, hist.prof_true_vs_reco_theta.GetName()
    )
    out_file.WriteObject(
        hist.prof_true_vs_reco_phi, hist.prof_true_vs_reco_phi.GetName()
    )

    out_file.WriteObject(
        hist.prof_theta_rel_resolution, hist.prof_theta_rel_resolution.GetName()
    )
    out_file.WriteObject(
        hist.prof_phi_rel_resolution, hist.prof_phi_rel_resolution.GetName()
    )

    out_file.WriteObject(
        hist.prof_theta_resolution, hist.prof_theta_resolution.GetName()
    )
    out_file.WriteObject(hist.prof_phi_resolution, hist.prof_phi_resolution.GetName())

    for element in detector.elements:
        out_file.WriteObject(
            hist.path_length(detector, element),
            hist.path_length(detector, element).GetName(),
        )

    plotter.save_canvases(out_file, output_directory, save_eps=True)


if __name__ == "__main__":
    # detector.plot(True, True, True, False, False)

    # -------------------------------
    # Running Metropolis-Hastings
    # -------------------------------
    muons = mcmc.metropolis_hastings(
        flux_model, initial_sample_guess, num_samples, proposal_std, burning
    )

    print("Size of muons array:", sys.getsizeof(muons))
    print("Size of muons array (nbytes):", muons.nbytes)

    # -------------------------------
    # Running muon loop
    # -------------------------------
    muon_loop(muons, detector, clear_muons)

    # -------------------------------
    # Running plots
    # -------------------------------
    make_plots(detector, file_name, output_directory)
