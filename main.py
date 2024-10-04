"""Muon flux simulation including detector studies."""
import os
import sys
import math
import numpy as np
import ROOT as R

from tqdm import tqdm

from muonsim import mc
from muonsim import muonflux
from muonsim import utils
from muonsim import geometry as GEO
from muonsim import histograms as HISTS
from muonsim import plotter
from muonsim.Detector import Detector
from muonsim.Generator import Generator

# -------------------------------
# Setting up muon generation.
# -------------------------------
# N_SAMPLES = 5_000_000
N_SAMPLES = 1000
PHI_RANGE = []
THETA_RANGE = []
np.random.seed(42)
# Muon generation will consider the geometry specified by the following
# sequences of modules.
REQUIRED_CONNECTIONS = ["TU1_TL1_BU5_BL5"]

# -------------------------------
# Setting up muon reconstruction.
# -------------------------------
# ... one always needs the connections ...
# DETECTOR = Detector(GEO.block_telescope.detector)
# DETECTOR = Detector(GEO.strip_telescope.detector, GEO.strip_telescope.connections)
DETECTOR = Detector(
    GEO.barmod_telescope.detector,
    GEO.barmod_telescope.connections,
    GEO.barmod_telescope.areas,
)

# DETECTOR = Detector(GEO.barmod_telescope_v2.detector, GEO.barmod_telescope_v2.connections)
# DETECTOR = Detector(GEO.sabre_telescope.detector, GEO.sabre_telescope.connections)

# Individual reconstructed events are displayed one by one.
# Muon parameters are printed on the screen.
LIVE_MODE = False

# Maximum amount of muons in memory.
CLEAR_MUONS = 10000

# Require coincidence of these specific modules. N.B. this is not the same
# as requiring specific coincidences, as events generated for one connection
# might hit modules outside of that connection, for example, if the generated
# angles are not constrained.
REQUIRED_MODULES = []

# Require that the muon intersects a certain number
# of upper and lower boundary planes. Every number
# in the list is a required number in OR with the others.
# The most inclusive requirement is 0, 1, 2.
REQUIRED_BOUNDARY_COINCIDENCES = [0, 1, 2]

# -------------------------------
# Setting up plotting
# -------------------------------
FILE_NAME = f"TestCosterfieldEffMap_{N_SAMPLES}_{GEO.strip_telescope.n_sensors}x{GEO.strip_telescope.n_sensors}_{GEO.strip_telescope.name}.root"
# FILE_NAME = f"BarMODType1_EffMap_{N_SAMPLES}_{GEO.strip_telescope.n_sensors}x{GEO.strip_telescope.n_sensors}_{GEO.strip_telescope.name}.root"
# FILE_NAME = f"BarMODType2_EffMap_{N_SAMPLES}_{GEO.strip_telescope.n_sensors}x{GEO.strip_telescope.n_sensors}_{GEO.strip_telescope.name}.root"
# FILE_NAME = f"SABREFlux_{N_SAMPLES}_{GEO.sabre_telescope.name}.root"
# FILE_NAME = f"Test_{N_SAMPLES}.root"
OUTPUT_DIRECTORY = FILE_NAME.split(".")[0]

# An interactive plot is displayed including not more than CLEAR_MUONS muons
# reconstructed by the detector.
SHOW_FINAL_PLOT = True


def muon_generation(
    detector, connections=[], n_samples=1000, phi_range=[], theta_range=[]
):
    """Generating muons."""

    if not connections:
        connections = detector.connections.keys()

    muons = {c: [] for c in connections}

    n_connections = len(connections)

    print(
        f"Generating {n_samples} muons for each one of {n_connections} connections ..."
    )
    for c_idx, connection in tqdm(
        enumerate(connections), total=n_connections, colour="green"
    ):
        generator = Generator(detector, connection, phi_range, theta_range)
        generator.generate_muons(n_samples)

        muons[connection] = generator.get_muons()

    return muons


def muon_reconstruction(
    all_muons,
    detector,
    histograms,
    boundary_coincidences=[2],
    event_modules=[],
    clear_muons=1000,
    live_mode=False,
):
    """Loop over generated muons. There is a different list
    for every generated connection. Histograms will be filled with all events
    reconstructed. N.B. histograms are created in a separate module."""

    for connection, muons in all_muons.items():
        n_gen_muons = len(muons)

        print(
            f"Reconstructing {n_gen_muons} generated muons for connection {connection} ..."
        )

        for m_idx, muon in tqdm(enumerate(muons), total=n_gen_muons, colour="red"):
            if m_idx % clear_muons == 0:
                detector.clear_muons(clear_muons)

            # Properties of the generated muon.
            # ----------------------------------------
            # Here units are in degrees.
            gen_true_theta = muon["theta"]
            gen_true_phi = muon["phi"]
            gen_true_origin = muon["start"]

            # Clearing all event-related data structures.
            detector.reset_event()

            # This is loading the muon event into memory.
            has_intersection = detector.intersect(
                gen_true_theta,
                gen_true_phi,
                gen_true_origin,
                required_boundary_coincidences=boundary_coincidences,
                required_modules=event_modules,
            )

            # Reconstruct the muon trajectory.
            has_reconstruction = detector.reconstruct()

            if has_reconstruction:
                if not has_intersection:
                    sys.exit(
                        "ERROR: the muon is reconstructed but has no intersections!"
                    )

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

                    histograms.path_length(detector, element).Fill(path)

                phi_rel_resolution = 0
                phi_resolution = reco_muon_phi - true_muon_phi

                if true_muon_phi > 0:
                    phi_rel_resolution = abs(phi_resolution) / true_muon_phi

                theta_rel_resolution = 0
                theta_resolution = reco_muon_theta - true_muon_theta

                if true_muon_theta > 0:
                    theta_rel_resolution = abs(theta_resolution) / true_muon_theta

                # Filling 1D histograms for all recorded muons.
                # ----------------------------------------------

                # Muon parameters true vs reco.
                histograms.true_theta.Fill(true_muon_theta)
                histograms.reco_theta.Fill(reco_muon_theta)

                # Muon path in the detector.
                histograms.total_path_length.Fill(total_path_length)
                histograms.top_path_length.Fill(top_path_length)
                histograms.bottom_path_length.Fill(bottom_path_length)

                # Filling 2D histograms for all recorded muons.
                # ----------------------------------------------

                # True vs reco histograms.
                histograms.true_vs_reco_theta.Fill(true_muon_theta, reco_muon_theta)
                histograms.true_vs_reco_phi.Fill(true_muon_phi, reco_muon_phi)

                # True vs reco profiles.
                histograms.prof_true_vs_reco_theta.Fill(
                    true_muon_theta, reco_muon_theta
                )
                histograms.prof_true_vs_reco_phi.Fill(true_muon_phi, reco_muon_phi)

                # Resolutions profiles.
                histograms.prof_theta_resolution.Fill(true_muon_theta, theta_resolution)
                histograms.prof_phi_resolution.Fill(true_muon_phi, phi_resolution)

                # Relative resolutions profiles.
                histograms.prof_theta_rel_resolution.Fill(
                    true_muon_theta, theta_rel_resolution
                )
                histograms.prof_phi_rel_resolution.Fill(
                    true_muon_phi, phi_rel_resolution
                )

                # Angular distance profile.
                histograms.prof2d_ang_dist_vs_true.Fill(
                    true_muon_theta, true_muon_phi, ang_dist
                )
                histograms.prof2d_ang_dist_vs_reco.Fill(
                    reco_muon_theta, reco_muon_phi, ang_dist
                )

                histograms.prof2d_ang_acc_vs_reco.Fill(
                    reco_muon_theta, reco_muon_phi, detector_acceptance / n_gen_muons
                )

            # Filling 1D histograms for all generated muons.
            # ----------------------------------------------
            # Generated parameters. To be filled outside the accepted muons loop.
            # histograms.energy.Fill(gen_true_energy)
            histograms.cos_theta.Fill(np.cos(np.pi * gen_true_theta / 180.0))

            # Filling 2D histograms for all generated muons.
            # ----------------------------------------------
            # Angular acceptance profile.
            histograms.prof2d_ang_acc_vs_true.Fill(
                gen_true_theta, gen_true_phi, detector_acceptance
            )


def make_plots(
    detector, file_name, histograms, output_directory=None, show_final_plot=True
):
    """Plotting. It includes displaying the detector and muon rays."""

    if show_final_plot:
        detector.plot(
            add_elements=True,
            add_muons=True,
            add_intersections=False,
            add_connections=False,
        )

    out_dir = "./"
    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        out_dir = output_directory

    file_name = os.path.join(out_dir, file_name)

    out_file = R.TFile.Open(file_name, "RECREATE")

    # out_file.WriteObject(histograms.energy, histograms.energy.GetName())
    out_file.WriteObject(histograms.cos_theta, histograms.cos_theta.GetName())

    out_file.WriteObject(histograms.true_theta, histograms.true_theta.GetName())
    out_file.WriteObject(histograms.reco_theta, histograms.reco_theta.GetName())

    out_file.WriteObject(
        histograms.total_path_length, histograms.total_path_length.GetName()
    )
    out_file.WriteObject(
        histograms.top_path_length, histograms.top_path_length.GetName()
    )
    out_file.WriteObject(
        histograms.bottom_path_length, histograms.bottom_path_length.GetName()
    )

    out_file.WriteObject(
        histograms.true_vs_reco_theta, histograms.true_vs_reco_theta.GetName()
    )
    out_file.WriteObject(
        histograms.true_vs_reco_phi, histograms.true_vs_reco_phi.GetName()
    )

    out_file.WriteObject(
        histograms.prof_true_vs_reco_theta, histograms.prof_true_vs_reco_theta.GetName()
    )
    out_file.WriteObject(
        histograms.prof_true_vs_reco_phi, histograms.prof_true_vs_reco_phi.GetName()
    )

    out_file.WriteObject(
        histograms.prof_theta_rel_resolution,
        histograms.prof_theta_rel_resolution.GetName(),
    )
    out_file.WriteObject(
        histograms.prof_phi_rel_resolution, histograms.prof_phi_rel_resolution.GetName()
    )

    out_file.WriteObject(
        histograms.prof_theta_resolution, histograms.prof_theta_resolution.GetName()
    )
    out_file.WriteObject(
        histograms.prof_phi_resolution, histograms.prof_phi_resolution.GetName()
    )

    out_file.WriteObject(
        histograms.prof2d_ang_dist_vs_true, histograms.prof2d_ang_dist_vs_true.GetName()
    )
    out_file.WriteObject(
        histograms.prof2d_ang_dist_vs_reco, histograms.prof2d_ang_dist_vs_reco.GetName()
    )
    out_file.WriteObject(
        histograms.prof2d_ang_acc_vs_true, histograms.prof2d_ang_acc_vs_true.GetName()
    )
    out_file.WriteObject(
        histograms.prof2d_ang_acc_vs_reco, histograms.prof2d_ang_acc_vs_reco.GetName()
    )

    for element in detector.elements:
        out_file.WriteObject(
            histograms.path_length(detector, element),
            histograms.path_length(detector, element).GetName(),
        )

    plotter.save_canvases(out_file, out_dir, save_eps=True)


if __name__ == "__main__":
    # -------------------------------
    # Generating muons
    # -------------------------------
    all_muons = muon_generation(
        detector=DETECTOR,
        connections=REQUIRED_CONNECTIONS,
        n_samples=N_SAMPLES,
        phi_range=PHI_RANGE,
        theta_range=THETA_RANGE,
    )

    print("Size of muons list:", sys.getsizeof(all_muons))
    # print("Size of muons array (nbytes):", muons.nbytes)

    # -------------------------------
    # Reconstructing muons.
    # -------------------------------
    muon_reconstruction(
        all_muons=all_muons,
        detector=DETECTOR,
        histograms=HISTS,
        boundary_coincidences=REQUIRED_BOUNDARY_COINCIDENCES,
        event_modules=REQUIRED_MODULES,
        clear_muons=CLEAR_MUONS,
        live_mode=LIVE_MODE,
    )

    # -------------------------------
    # Writing plots to file.
    # -------------------------------
    make_plots(
        detector=DETECTOR,
        file_name=FILE_NAME,
        histograms=HISTS,
        output_directory=OUTPUT_DIRECTORY,
        show_final_plot=SHOW_FINAL_PLOT,
    )
