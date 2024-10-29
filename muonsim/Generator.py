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

            _start = self.detector.generate_hit(self.connection, "top")
            _stop = self.detector.generate_hit(self.connection, "bottom")

            start = np.array(_start)
            stop = np.array(_stop)

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
