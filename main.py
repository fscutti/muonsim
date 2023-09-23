from muonsim import mcmc
from muonsim import muonflux

# from muonsim import testflux
from muonsim import utils
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

live_mode = False

# Initial guess for the sample and number of MCMC samples
initial_sample_guess = np.array([1.0, 1.0])
proposal_std = [0.01, 0.5]
# num_samples = 5_000_000
num_samples = 100_000
burning = int(num_samples * 0.30)
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

    if muons is None:
        print("Generating uniform muon distribution")

        muons = []
        for m_idx in range(num_samples):
            theta = np.random.uniform(0.0, 30.0)
            energy = np.random.uniform(0, 1000)
            cos_theta = np.cos(np.pi * theta / 180.0)

            muons.append([cos_theta, energy])

    for m_idx, m in tqdm(enumerate(muons), total=len(muons), colour="red"):
        if m_idx % clear_muons == 0:
            detector.clear_muons(clear_muons)

        cos_theta, energy = m

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
        has_intersection = detector.intersect(
            muon_true_theta,
            muon_true_phi,
            muon_true_origin,
            coincidences=coincidences,
            event_modules=event_modules,
        )

        # Reconstruct the muon trajectory.
        has_reconstruction = detector.reconstruct()

        total_path_length = 0
        top_path_length = 0
        bottom_path_length = 0

        # hist.true_theta.Fill(muon_true_theta)
        # hist.cos_theta.Fill(cos_theta)
        # hist.energy.Fill(energy)

        # To be counted muons have to have intersections with the
        # detector volume boundaries and a reconstructed muon has
        # to be found.
        if has_intersection and has_reconstruction:
            muon_hits = detector.get_muon_hits()

            true_muon, reco_muon = detector.get_muon_event("angles")
            v_true_muon, v_reco_muon = detector.get_muon_event("endpoints")

            true_start, true_stop = v_true_muon
            reco_start, reco_stop = v_reco_muon

            true_direction = true_start - true_stop
            reco_direction = reco_start - reco_stop

            true_muon_theta, true_muon_phi = true_muon
            reco_muon_theta, reco_muon_phi = reco_muon

            ang_dist = utils.angular_distance_3d(reco_direction, true_direction)

            if live_mode:
                print("----------------------------------------")
                print("Theta (reco, true): ", reco_muon_theta, true_muon_theta)
                print("Phi (reco, true): ", reco_muon_phi, true_muon_phi)
                print("Angular distance (reco, true):", ang_dist)
                print("----------------------------------------")
                print()

                detector.plot(
                    add_elements=True,
                    add_muons=False,
                    add_intersections=False,
                    add_connections=False,
                    start=[v_true_muon[0], v_reco_muon[0]],
                    stop=[v_true_muon[1], v_reco_muon[1]],
                )

            # Filling histograms for individual elements.
            for element, hits in muon_hits.items():
                start, stop = hits
                path = np.linalg.norm(start - stop)

                total_path_length += path

                if element.startswith("T"):
                    top_path_length += path

                elif element.startswith("B"):
                    bottom_path_length += path

                hist.path_length(detector, element).Fill(path)

            phi_rel_resolution = 0
            phi_resolution = reco_muon_phi - true_muon_phi

            if true_muon_phi > 0:
                phi_rel_resolution = abs(phi_resolution) / true_muon_phi

            theta_rel_resolution = 0
            theta_resolution = reco_muon_theta - true_muon_theta

            if true_muon_theta > 0:
                theta_rel_resolution = abs(theta_resolution) / true_muon_theta

            # Filling 1D histograms for all generated muons.
            # ----------------------------------------------

            # Generated parameters.
            hist.energy.Fill(energy)
            hist.cos_theta.Fill(cos_theta)

            # Muon parameters true vs reco.
            hist.true_theta.Fill(true_muon_theta)
            hist.reco_theta.Fill(reco_muon_theta)

            # Muon path in the detector.
            hist.total_path_length.Fill(total_path_length)
            hist.top_path_length.Fill(top_path_length)
            hist.bottom_path_length.Fill(bottom_path_length)

            # Filling 2D histograms for all generated muons.
            # ----------------------------------------------

            # True vs reco histograms.
            hist.true_vs_reco_theta.Fill(true_muon_theta, reco_muon_theta)
            hist.true_vs_reco_phi.Fill(true_muon_phi, reco_muon_phi)

            # True vs reco profiles.
            hist.prof_true_vs_reco_theta.Fill(true_muon_theta, reco_muon_theta)
            hist.prof_true_vs_reco_phi.Fill(true_muon_phi, reco_muon_phi)

            # Resolutions profiles.
            hist.prof_theta_resolution.Fill(true_muon_theta, theta_resolution)
            hist.prof_phi_resolution.Fill(true_muon_phi, phi_resolution)

            # Relative resolutions profiles.
            hist.prof_theta_rel_resolution.Fill(true_muon_theta, theta_rel_resolution)
            hist.prof_phi_rel_resolution.Fill(true_muon_phi, phi_rel_resolution)

            # Angular distance profile.
            hist.prof2d_ang_dist.Fill(true_muon_theta, true_muon_phi, ang_dist)


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

    out_file.WriteObject(hist.prof2d_ang_dist, hist.prof2d_ang_dist.GetName())

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
    # muons = mcmc.metropolis_hastings(
    #    flux_model, initial_sample_guess, num_samples, proposal_std, burning
    # )

    # print("Size of muons array:", sys.getsizeof(muons))
    # print("Size of muons array (nbytes):", muons.nbytes)
    muons = None
    # -------------------------------
    # Running muon loop
    # -------------------------------
    muon_loop(muons, detector, clear_muons)

    # -------------------------------
    # Running plots
    # -------------------------------
    make_plots(detector, file_name, output_directory)
