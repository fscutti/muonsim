from muonsim import mcmc
from muonsim import muonflux
from muonsim import geometry as geo
from muonsim import histograms as hist

from muonsim.Detector import Detector

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
num_samples = 5_000_000
# num_samples = 1000000
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


def muon_loop(muons, detector, clear_muons=1000):
    """Loop over generated muons."""

    for m_idx, m in tqdm(enumerate(muons), total=len(muons), colour="red"):
        cos_theta, energy = m

        if m_idx % clear_muons == 0:
            detector.clear_muons(clear_muons)

        # Here units are in degrees.
        muon_theta = 180.0 * math.acos(cos_theta) / np.pi
        muon_phi = np.random.uniform(0.0, 360.0)

        # Generating the muon on the top panel.
        muon_x = np.random.uniform(*detector.volume["x"])
        muon_y = np.random.uniform(*detector.volume["y"])
        muon_z = max(detector.volume["z"])

        muon_origin = np.array([muon_x, muon_y, muon_z])

        # Clearing all event-related data structures.
        detector.reset_event()

        # This is loading the muon event into memory.
        event_points = detector.intersect(
            muon_theta,
            muon_phi,
            muon_origin,
            coincidences=coincidences,
            event_modules=event_modules,
        )

        event_points = detector.get_event_points()
        event_reconstruction = detector.get_event_reconstruction()

        total_path_length = 0
        top_path_length = 0
        bottom_path_length = 0

        # if the event satisfies all our selection criteria.
        if event_points and event_reconstruction:
            # Filling histograms for individual elements.
            for element, points in event_points.items():
                start, stop = points

                path = np.linalg.norm(start - stop)
                total_path_length += path
                hist.path_length[element].Fill(path)

                if element.startswith("T"):
                    top_path_length += path

                elif element.startswith("B"):
                    bottom_path_length += path

            # Filling histograms for all generated muons.
            hist.energy.Fill(energy)
            hist.cos_theta.Fill(cos_theta)

            hist.total_path_length.Fill(total_path_length)
            hist.top_path_length.Fill(top_path_length)
            hist.bottom_path_length.Fill(bottom_path_length)


def make_plots(detector, file_name):
    """Plotting. It includes displaying the detector and muon rays."""

    detector.plot(
        add_elements=True, add_muons=True, add_intersections=True, add_connections=False
    )

    out_file = R.TFile.Open(file_name, "RECREATE")

    out_file.WriteObject(hist.energy, hist.energy.GetName())
    out_file.WriteObject(hist.cos_theta, hist.cos_theta.GetName())
    out_file.WriteObject(hist.total_path_length, hist.total_path_length.GetName())
    out_file.WriteObject(hist.top_path_length, hist.top_path_length.GetName())
    out_file.WriteObject(hist.bottom_path_length, hist.bottom_path_length.GetName())

    for element in detector.elements:
        out_file.WriteObject(
            hist.path_length[element], hist.path_length[element].GetName()
        )


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
    make_plots(detector, file_name)
