"""Muon flux simulation including detector studies."""
import os
import math
import sys
import numpy as np
import ROOT as R

from tqdm import tqdm

from muonsim import mc
from muonsim import muonflux
from muonsim import utils
from muonsim import geometry as geo
from muonsim import histograms as hist
from muonsim import plotter
from muonsim.Detector import Detector

LIVE_MODE = False

# -------------------------------
# Setting up Metropolis-Hastings
# -------------------------------
# If MCMC has to be used then MUONS should be an empty list.
# NUM_SAMPLES = 5_000_000
NUM_SAMPLES = 100000
USE_MUON_FLUX = False
MIN_PHI, MAX_PHI = 0.0, 360.0

if USE_MUON_FLUX:
    np.random.seed(42)
    # Initial guess for the sample and number of MCMC samples
    INITIAL_SAMPLE_GUESS = np.array([1.0, 1.0])
    PROPOSAL_STD = [0.01, 0.5]
    BURNING = int(NUM_SAMPLES * 0.30)
    FLUX_MODEL = muonflux.sea_level
else:
    # These parameters are only used for the uniform distribution.
    MIN_THETA, MAX_THETA = 0.0, 20.0
    MIN_ENERGY, MAX_ENERGY = 0.0, 1000.0
    RANGE = {
        "theta": [MIN_THETA, MAX_THETA],
        "energy": [MIN_ENERGY, MAX_ENERGY],
    }

# -------------------------------
# Setting up muon loop
# -------------------------------
# detector = Detector(geo.block_telescope.detector)
DETECTOR = Detector(geo.strip_telescope.detector, geo.strip_telescope.connections)

# Maximum amount of muons in memory.
CLEAR_MUONS = 1000
# Require coincidence of these specific modules.
# required_modules = ["T12", "B12"]
REQUIRED_MODULES = []
REQUIRED_COINCIDENCES = [2]

# -------------------------------
# Setting up plotting
# -------------------------------
FILE_NAME = f"MuonSimTest_{NUM_SAMPLES}_{geo.block_telescope.n_sensors}x{geo.block_telescope.n_sensors}_all_sensors.root"
OUTPUT_DIRECTORY = FILE_NAME.split(".")[0]


def muon_loop(muons, detector, clear_muons=1000):
    """Loop over generated muons."""
    for m_idx, muon in tqdm(enumerate(muons), total=len(muons), colour="red"):
        if m_idx % clear_muons == 0:
            detector.clear_muons(clear_muons)

        theta, energy = muon

        # Geometrical properties of the true muon.
        # ----------------------------------------
        # Here units are in degrees.
        muon_true_theta = theta
        muon_true_phi = np.random.uniform(MIN_PHI, MAX_PHI)

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
            required_coincidences=REQUIRED_COINCIDENCES,
            required_modules=REQUIRED_MODULES,
        )

        # Reconstruct the muon trajectory.
        has_reconstruction = detector.reconstruct()

        # print(has_reconstruction, has_intersection)

        if has_reconstruction:
            if not has_intersection:
                sys.exit("ERROR: the muon is reconstructed but has no intersections!")

        total_path_length = 0
        top_path_length = 0
        bottom_path_length = 0

        detector_acceptance = 1 if has_reconstruction else 0

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

            if LIVE_MODE:
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
            hist.cos_theta.Fill(np.cos(np.pi * theta / 180.0))

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

        # Angular acceptance profile. Has to be filled outside the condition statement.
        hist.prof2d_ang_acc.Fill(muon_true_theta, muon_true_phi, detector_acceptance)


def make_plots(detector, file_name, output_directory=None):
    """Plotting. It includes displaying the detector and muon rays."""

    # detector.plot(
    #    add_elements=True, add_muons=True, add_intersections=True, add_connections=False
    # )
    out_dir = "./"
    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        out_dir = output_directory

    file_name = os.path.join(out_dir, file_name)

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
    out_file.WriteObject(hist.prof2d_ang_acc, hist.prof2d_ang_acc.GetName())

    for element in detector.elements:
        out_file.WriteObject(
            hist.path_length(detector, element),
            hist.path_length(detector, element).GetName(),
        )

    plotter.save_canvases(out_file, out_dir, save_eps=True)


if __name__ == "__main__":
    # detector.plot(True, True, True, False, False)

    # -------------------------------
    # Generating muons
    # -------------------------------
    muons = None

    if USE_MUON_FLUX:
        muons = mc.metropolis_hastings(
            FLUX_MODEL, INITIAL_SAMPLE_GUESS, NUM_SAMPLES, PROPOSAL_STD, BURNING
        )
    else:
        muons = mc.uniform(NUM_SAMPLES, RANGE)

    if muons is None:
        sys.exit("ERROR: no available muons!")

    print("Size of muons array:", sys.getsizeof(muons))
    print("Size of muons array (nbytes):", muons.nbytes)

    # -------------------------------
    # Running muon loop
    # -------------------------------
    muon_loop(muons, DETECTOR, CLEAR_MUONS)

    # -------------------------------
    # Running plots
    # -------------------------------
    make_plots(DETECTOR, FILE_NAME, OUTPUT_DIRECTORY)
