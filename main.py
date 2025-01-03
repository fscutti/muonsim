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
WALKING_HITS = True
# N_SAMPLES = 5_000_000
N_SAMPLES = 20
#PHI_RANGE = [0, 360]
#THETA_RANGE = [0, 60]  # Used for BarMOD2
# THETA_RANGE = [0, 20] # Used for BarMOD1
PHI_RANGE = []
THETA_RANGE = []
np.random.seed(42)
# Muon generation will consider the geometry specified by the following
# sequences of modules.
#REQUIRED_CONNECTIONS = []
REQUIRED_CONNECTIONS = ["TU3_TL3_BU3_BL3"]

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
CLEAR_MUONS = 1000

# Require coincidence of these specific modules. N.B. this is not the same
# as requiring specific coincidences, as events generated for one connection
# might hit modules outside of that connection, for example, if the generated
# angles are not constrained.
REQUIRED_MODULES = []
REQUIRED_N_MODULES = [4, 5, 6, 7, 8]
# REQUIRED_N_MODULES = [4]

# Require that the muon intersects a certain number
# of upper and lower boundary planes. Every number
# in the list is a required number in OR with the others.
# The most inclusive requirement is 0, 1, 2.
# IMPORTANT: boundary planes requirements are not the same
# as hit requirements. A muon can have a meaningful number of hits
# in the detector while not intersecting any boundary surfaces,
# e.g. if it comes from the side of the detector volume.
REQUIRED_BOUNDARY_COINCIDENCES = [0, 1, 2]

# -------------------------------
# Setting up plotting
# -------------------------------
# FILE_NAME = f"BinningCosterfieldEffMap_{N_SAMPLES}_{GEO.strip_telescope.n_sensors}x{GEO.strip_telescope.n_sensors}_{GEO.strip_telescope.name}.root"
# FILE_NAME = f"NewBarMODType1_EffMap_{N_SAMPLES}_{GEO.strip_telescope.n_sensors}x{GEO.strip_telescope.n_sensors}_{GEO.strip_telescope.name}.root"
# FILE_NAME = f"NewBarMODType2_EffMap_{N_SAMPLES}_{GEO.strip_telescope.n_sensors}x{GEO.strip_telescope.n_sensors}_{GEO.strip_telescope.name}.root"
# FILE_NAME = f"SABREFlux_{N_SAMPLES}_{GEO.sabre_telescope.name}.root"
# FILE_NAME = f"Test_{N_SAMPLES}.root"

FILE_NAME = f"26Nov2024_TEST_BarMODType2_EffMap_{N_SAMPLES}_{GEO.strip_telescope.n_sensors}x{GEO.strip_telescope.n_sensors}_{GEO.strip_telescope.name}.root"
#FILE_NAME = f"18Nov2024_PDF_BarMODType2_EffMap_{N_SAMPLES}_{GEO.strip_telescope.n_sensors}x{GEO.strip_telescope.n_sensors}_{GEO.strip_telescope.name}.root"
OUTPUT_DIRECTORY = FILE_NAME.split(".")[0]

#CORRECTION_FILE_NAME = f"/Users/fscutti/github/muonsim/12Nov2024_BarMODType2_EffMap_5000_5x5_Strip/12Nov2024_BarMODType2_EffMap_5000_5x5_Strip.root"
#CORRECTION_FILE_NAME = f"/Users/fscutti/github/muonsim/14Nov2024_BarMODType2_EffMap_5000_5x5_Strip/14Nov2024_BarMODType2_EffMap_5000_5x5_Strip.root"
#CORRECTION_FILE_NAME = f"/Users/fscutti/github/muonsim/18Nov2024_PDF_BarMODType2_EffMap_10000_5x5_Strip/18Nov2024_PDF_BarMODType2_EffMap_10000_5x5_Strip.root"
CORRECTION_FILE_NAME = f"/Users/fscutti/github/muonsim/18Nov2024_PDF_BarMODType2_EffMap_50000_5x5_Strip/18Nov2024_PDF_BarMODType2_EffMap_50000_5x5_Strip.root"

corr_file = R.TFile.Open(CORRECTION_FILE_NAME, "READ")

H_PDF = None
#H_PDF = corr_file.Get("h2_reco_correction")
H_ACC = corr_file.Get("p2_ang_acc_vs_true_high_reso")
#H_ACC = corr_file.Get("p2_ang_acc_vs_true")
#H_ACC = corr_file.Get("p2_ang_acc_vs_true_low_reso")

# An interactive plot is displayed including not more than CLEAR_MUONS muons
# reconstructed by the detector.
SHOW_FINAL_PLOT = False


def muon_generation(
    detector, connections=[], n_samples=1000, phi_range=[], theta_range=[], walking_hits=False
):
    """Generating muons."""

    if not connections:
        connections = detector.connections.keys()

    muons = {c: [] for c in connections}

    n_connections = len(connections)

    for c_idx, connection in tqdm(
        enumerate(connections), total=n_connections, colour="green"
    ):
        generator = Generator(detector, connection, phi_range, theta_range, walking_hits)
        generator.generate_muons(n_samples)

        muons[connection] = generator.get_muons()

    return muons


def muon_reconstruction(
    all_muons,
    detector,
    histograms,
    boundary_coincidences=[2],
    event_modules=[],
    event_n_modules=[],
    clear_muons=1000,
    live_mode=False,
):
    """Loop over generated muons. There is a different list
    for every generated connection. Histograms will be filled with all events
    reconstructed. N.B. histograms are created in a separate module."""

    # min_norm = float("inf")

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
                required_n_modules=event_n_modules,
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

                # min_norm = min(min_norm, np.linalg.norm(reco_direction))

                true_muon_theta, true_muon_phi = true_muon
                reco_muon_theta, reco_muon_phi = reco_muon

                true_muon_elevation = 90.0 - true_muon_theta
                reco_muon_elevation = 90.0 - reco_muon_theta

                ang_dist = utils.angular_distance_3d(reco_direction, true_direction)

                el_dist = 2.0 * utils.angle_distance(
                    reco_muon_elevation, true_muon_elevation
                )
                az_dist = 2.0 * utils.angle_distance(reco_muon_phi, true_muon_phi)

                # sq_el_diff = el_dist**2
                # sq_phi_diff = az_dist**2

                if live_mode:
                    print("----------------------------------------")
                    print("Theta (reco, true): ", reco_muon_theta, true_muon_theta)
                    print("Phi (reco, true): ", reco_muon_phi, true_muon_phi)
                    print("Angular distance (reco, true):", ang_dist)
                    print("Elevation distance (reco, true):", el_dist)
                    print("Azimuthal distance (reco, true):", az_dist)
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
                # for element, hits in muon_hits.items():
                #    start, stop = hits
                #    path = np.linalg.norm(start - stop)

                #    total_path_length += path

                #    if element.startswith("T"):
                #        top_path_length += path

                #    elif element.startswith("B"):
                #        bottom_path_length += path

                #    histograms.path_length(detector, element).Fill(path)

                # phi_rel_resolution = 0
                # phi_resolution = reco_muon_phi - true_muon_phi

                # if true_muon_phi > 0:
                #    phi_rel_resolution = abs(phi_resolution) / true_muon_phi

                # theta_rel_resolution = 0
                # theta_resolution = reco_muon_theta - true_muon_theta

                # if true_muon_theta > 0:
                #    theta_rel_resolution = abs(theta_resolution) / true_muon_theta

                # Filling 1D histograms for all recorded muons.
                # ----------------------------------------------

                # Muon parameters true vs reco.
                histograms.true_theta.Fill(true_muon_theta)
                histograms.reco_theta.Fill(reco_muon_theta)

                # Muon path in the detector.
                # histograms.total_path_length.Fill(total_path_length)
                # histograms.top_path_length.Fill(top_path_length)
                # histograms.bottom_path_length.Fill(bottom_path_length)

                # Filling 2D histograms for all recorded muons.
                # ----------------------------------------------

                histograms.reco_raw.Fill(reco_muon_theta, reco_muon_phi, 1)
                histograms.true_raw.Fill(true_muon_theta, true_muon_phi, 1)

                # Reconstructed muons coordinates.
                acceptance_weight_reco = utils.get_corr_weight(
                    H_ACC, reco_muon_theta, reco_muon_phi
                )
                acceptance_weight_true = utils.get_corr_weight(
                    H_ACC, true_muon_theta, true_muon_phi
                )
                pdf_weight_reco = utils.get_corr_weight(
                    H_PDF, reco_muon_theta, reco_muon_phi
                )

                if acceptance_weight_true <= 0:
                    print("WARNING: acceptance weights for true muons should be always positive!")

                if acceptance_weight_true:
                    # True muons coordinates.
                    histograms.true_weighted.Fill(
                        true_muon_theta, true_muon_phi, 1.0 / acceptance_weight_true
                    )
                    histograms.reco_weighted.Fill(
                        reco_muon_theta,
                        reco_muon_phi,
                        1.0
                        / acceptance_weight_true,  # testing with the true muon weight
                        #/ acceptance_weight_reco,  # testing with the reco muon weight
                    )
                
                if acceptance_weight_reco:
                    histograms.reco_validation.Fill(
                        reco_muon_theta,
                        reco_muon_phi,
                        pdf_weight_reco / acceptance_weight_reco,
                    )

                # True vs reco histograms.
                # histograms.true_vs_reco_theta.Fill(true_muon_theta, reco_muon_theta)
                # histograms.true_vs_reco_phi.Fill(true_muon_phi, reco_muon_phi)

                # True vs reco profiles.
                # histograms.prof_true_vs_reco_theta.Fill(
                #    true_muon_theta, reco_muon_theta
                # )
                # histograms.prof_true_vs_reco_phi.Fill(true_muon_phi, reco_muon_phi)

                # Resolutions profiles.
                # histograms.prof_theta_resolution.Fill(true_muon_theta, theta_resolution)
                # histograms.prof_phi_resolution.Fill(true_muon_phi, phi_resolution)

                # Relative resolutions profiles.
                # histograms.prof_theta_rel_resolution.Fill(
                #    true_muon_theta, theta_rel_resolution
                # )
                # histograms.prof_phi_rel_resolution.Fill(
                #    true_muon_phi, phi_rel_resolution
                # )

                # Angular distance profile.
                # histograms.prof2d_ang_dist_vs_true.Fill(
                #    true_muon_theta, true_muon_phi, ang_dist
                # )
                # histograms.prof2d_ang_dist_vs_reco.Fill(
                #    reco_muon_theta, reco_muon_phi, ang_dist
                # )

                # RMS vs reco, RMS vs true.
                utils.fill_max(
                    histograms.el_max_dist_vs_reco,
                    reco_muon_elevation,
                    reco_muon_phi,
                    el_dist,
                )
                utils.fill_max(
                    histograms.el_max_dist_vs_true,
                    true_muon_elevation,
                    true_muon_phi,
                    el_dist,
                )

                utils.fill_max(
                    histograms.az_max_dist_vs_reco,
                    reco_muon_elevation,
                    reco_muon_phi,
                    az_dist,
                )
                utils.fill_max(
                    histograms.az_max_dist_vs_true,
                    true_muon_elevation,
                    true_muon_phi,
                    az_dist,
                )

                # histograms.prof2d_el_rms_vs_reco.Fill(
                #    reco_muon_elevation, reco_muon_phi, sq_el_diff
                # )
                # histograms.prof2d_el_rms_vs_true.Fill(
                #    true_muon_elevation, true_muon_phi, sq_el_diff
                # )

                # histograms.prof2d_az_rms_vs_reco.Fill(
                #    reco_muon_elevation, reco_muon_phi, sq_phi_diff
                # )
                # histograms.prof2d_az_rms_vs_true.Fill(
                #    true_muon_elevation, true_muon_phi, sq_phi_diff
                # )

            # Filling 1D histograms for all generated muons.
            # ----------------------------------------------
            # Generated parameters. To be filled outside the accepted muons loop.
            # histograms.energy.Fill(gen_true_energy)
            # histograms.cos_theta.Fill(np.cos(np.pi * gen_true_theta / 180.0))

            # Filling 2D histograms for all generated muons.
            # ----------------------------------------------
            # Angular acceptance profile.
            histograms.prof2d_ang_acc_vs_true_high_reso.Fill(
                gen_true_theta, gen_true_phi, detector_acceptance
            )
            histograms.prof2d_ang_acc_vs_true_low_reso.Fill(
                gen_true_theta, gen_true_phi, detector_acceptance
            )

            # Warning: the following histogram will just contain 1.
            # if detector_acceptance:
            #    histograms.prof2d_ang_acc_vs_reco.Fill(
            #        reco_muon_theta, reco_muon_phi, detector_acceptance
            #    )

    # print(min_norm)


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
    # out_file.WriteObject(histograms.cos_theta, histograms.cos_theta.GetName())

    out_file.WriteObject(histograms.true_theta, histograms.true_theta.GetName())
    out_file.WriteObject(histograms.reco_theta, histograms.reco_theta.GetName())

    # Normalise distributions.
    # histograms.true_raw = utils.make_inv_pdf(histograms.true_raw)
    # histograms.true_weighted = utils.make_inv_pdf(histograms.true_weighted)

    # histograms.reco_raw = utils.make_inv_pdf(histograms.reco_raw)
    # histograms.reco_weighted = utils.make_inv_pdf(histograms.reco_weighted)

    utils.make_pdf(
        histograms.reco_correction, histograms.reco_weighted, perform_inversion=True, normalise_angle=True
    )
    out_file.WriteObject(
        histograms.reco_correction, histograms.reco_correction.GetName()
    )
    
    utils.make_pdf(
        histograms.reco_validation, histograms.reco_validation, normalise_angle=True
    )
    out_file.WriteObject(
        histograms.reco_validation, histograms.reco_validation.GetName()
    )

    out_file.WriteObject(histograms.true_raw, histograms.true_raw.GetName())
    
    utils.make_pdf(
        histograms.true_weighted, histograms.true_weighted, normalise_angle=True
    )
    out_file.WriteObject(histograms.true_weighted, histograms.true_weighted.GetName())


    out_file.WriteObject(histograms.reco_raw, histograms.reco_raw.GetName())
    out_file.WriteObject(histograms.reco_weighted, histograms.reco_weighted.GetName())

    # out_file.WriteObject(
    #    histograms.total_path_length, histograms.total_path_length.GetName()
    # )
    # out_file.WriteObject(
    #    histograms.top_path_length, histograms.top_path_length.GetName()
    # )
    # out_file.WriteObject(
    #    histograms.bottom_path_length, histograms.bottom_path_length.GetName()
    # )

    # out_file.WriteObject(
    #    histograms.true_vs_reco_theta, histograms.true_vs_reco_theta.GetName()
    # )
    # out_file.WriteObject(
    #    histograms.true_vs_reco_phi, histograms.true_vs_reco_phi.GetName()
    # )

    # out_file.WriteObject(
    #    histograms.prof_true_vs_reco_theta, histograms.prof_true_vs_reco_theta.GetName()
    # )
    # out_file.WriteObject(
    #    histograms.prof_true_vs_reco_phi, histograms.prof_true_vs_reco_phi.GetName()
    # )

    # out_file.WriteObject(
    #    histograms.prof_theta_rel_resolution,
    #    histograms.prof_theta_rel_resolution.GetName(),
    # )
    # out_file.WriteObject(
    #    histograms.prof_phi_rel_resolution, histograms.prof_phi_rel_resolution.GetName()
    # )

    # out_file.WriteObject(
    #    histograms.prof_theta_resolution, histograms.prof_theta_resolution.GetName()
    # )
    # out_file.WriteObject(
    #    histograms.prof_phi_resolution, histograms.prof_phi_resolution.GetName()
    # )

    # out_file.WriteObject(
    #    histograms.prof2d_ang_dist_vs_true, histograms.prof2d_ang_dist_vs_true.GetName()
    # )
    # out_file.WriteObject(
    #    histograms.prof2d_ang_dist_vs_reco, histograms.prof2d_ang_dist_vs_reco.GetName()
    # )
    out_file.WriteObject(
        histograms.prof2d_ang_acc_vs_true_high_reso, histograms.prof2d_ang_acc_vs_true_high_reso.GetName()
    )
    out_file.WriteObject(
        histograms.prof2d_ang_acc_vs_true_low_reso, histograms.prof2d_ang_acc_vs_true_low_reso.GetName()
    )
    # out_file.WriteObject(
    #    histograms.prof2d_ang_acc_vs_reco, histograms.prof2d_ang_acc_vs_reco.GetName()
    # )

    # The following four histograms will need some transformation.

    # Transforming the TProfile2D objects in 2D histograms.
    # histograms.prof2d_el_rms_vs_reco = histograms.prof2d_el_rms_vs_reco.ProjectionXY()
    # histograms.prof2d_el_rms_vs_true = histograms.prof2d_el_rms_vs_true.ProjectionXY()
    # histograms.prof2d_az_rms_vs_reco = histograms.prof2d_az_rms_vs_reco.ProjectionXY()
    # histograms.prof2d_az_rms_vs_true = histograms.prof2d_az_rms_vs_true.ProjectionXY()

    # for glob_idx, el_idx, az_idx in utils.bin_loop(histograms.prof2d_el_rms_vs_reco):
    #    bin_content = histograms.prof2d_el_rms_vs_reco.GetBinContent(glob_idx)
    #    # bin_norm = utils.bin_solid_angle(histograms.prof2d_el_rms_vs_reco, el_idx, az_idx)
    #    histograms.prof2d_el_rms_vs_reco.SetBinContent(glob_idx, math.sqrt(bin_content))

    # for glob_idx, el_idx, az_idx in utils.bin_loop(histograms.prof2d_el_rms_vs_true):
    #    bin_content = histograms.prof2d_el_rms_vs_true.GetBinContent(glob_idx)
    #    # bin_norm = utils.bin_solid_angle(histograms.prof2d_el_rms_vs_true, el_idx, az_idx)
    #    histograms.prof2d_el_rms_vs_true.SetBinContent(glob_idx, math.sqrt(bin_content))

    # for glob_idx, el_idx, az_idx in utils.bin_loop(histograms.prof2d_az_rms_vs_reco):
    #    bin_content = histograms.prof2d_az_rms_vs_reco.GetBinContent(glob_idx)
    #    # bin_norm = utils.bin_solid_angle(histograms.prof2d_az_rms_vs_reco, el_idx, az_idx)
    #    histograms.prof2d_az_rms_vs_reco.SetBinContent(glob_idx, math.sqrt(bin_content))

    # for glob_idx, el_idx, az_idx in utils.bin_loop(histograms.prof2d_az_rms_vs_true):
    #    bin_content = histograms.prof2d_az_rms_vs_true.GetBinContent(glob_idx)
    #    # bin_norm = utils.bin_solid_angle(histograms.prof2d_az_rms_vs_true, el_idx, az_idx)
    #    histograms.prof2d_az_rms_vs_true.SetBinContent(glob_idx, math.sqrt(bin_content))

    # out_file.WriteObject(
    #    histograms.prof2d_el_rms_vs_reco, histograms.prof2d_el_rms_vs_reco.GetName()
    # )
    # out_file.WriteObject(
    #    histograms.prof2d_el_rms_vs_true, histograms.prof2d_el_rms_vs_true.GetName()
    # )
    # out_file.WriteObject(
    #    histograms.prof2d_az_rms_vs_reco, histograms.prof2d_az_rms_vs_reco.GetName()
    # )
    # out_file.WriteObject(
    #    histograms.prof2d_az_rms_vs_true, histograms.prof2d_az_rms_vs_true.GetName()
    # )

    out_file.WriteObject(
        histograms.el_max_dist_vs_reco, histograms.el_max_dist_vs_reco.GetName()
    )
    out_file.WriteObject(
        histograms.el_max_dist_vs_true, histograms.el_max_dist_vs_true.GetName()
    )
    out_file.WriteObject(
        histograms.az_max_dist_vs_reco, histograms.az_max_dist_vs_reco.GetName()
    )
    out_file.WriteObject(
        histograms.az_max_dist_vs_true, histograms.az_max_dist_vs_true.GetName()
    )

    # for element in detector.elements:
    #    out_file.WriteObject(
    #        histograms.path_length(detector, element),
    #        histograms.path_length(detector, element).GetName(),
    #    )

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
        walking_hits=WALKING_HITS,
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
        event_n_modules=REQUIRED_N_MODULES,
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
