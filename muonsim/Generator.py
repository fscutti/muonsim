"""This module generates muon trajectories."""
import sys

import numpy as np

from itertools import product

from muonsim import utils


class Generator:
    def __init__(self, detector, connection, phi_range=[], theta_range=[]):
        self.detector = detector
        self.connection = connection
        self.phi_range = phi_range
        self.theta_range = theta_range

        self._muons = []

        self._initialise()

    def _initialise(self):
        """Sanity checks."""
        if not self.connection in self.detector.connections:
            err_msg = f"ERROR: connection {self.connection} is not part of the "
            err_msg += "detector geometry!"
            sys.exit(err_msg)

    def get_muons(self):
        """Returning the generated muons list."""
        return self._muons

    def generate_muons(self, n_samples):
        """Filling the muons dictionary with n_samples generated particles."""
        for m_idx in range(n_samples):
            muon = {"start": None, "stop": None, "theta": None, "phi": None}

            start_range_x = self.detector.get_intersection_extension(
                self.connection, "top", "x"
            )
            start_range_y = self.detector.get_intersection_extension(
                self.connection, "top", "y"
            )

            stop_range_x = self.detector.get_intersection_extension(
                self.connection, "bottom", "x"
            )
            stop_range_y = self.detector.get_intersection_extension(
                self.connection, "bottom", "y"
            )

            start_x = np.random.uniform(*start_range_x)
            start_y = np.random.uniform(*start_range_y)

            stop_x = np.random.uniform(*stop_range_x)
            stop_y = np.random.uniform(*stop_range_y)

            start_z = self.detector.get_intersection_center(self.connection, "top", "z")
            stop_z = self.detector.get_intersection_center(
                self.connection, "bottom", "z"
            )

            start = np.array([start_x, start_y, start_z])
            stop = np.array([stop_x, stop_y, stop_z])

            theta, phi = utils.get_polar_coor(start, stop)

            if self.phi_range:
                phi = np.random.uniform(*self.phi_range)

            if self.theta_range:
                theta = np.random.uniform(*self.theta_range)

            muon["start"] = start
            muon["stop"] = stop

            muon["phi"] = phi
            muon["theta"] = theta

            self._muons.append(muon)


# EOF
