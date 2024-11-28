"""This module generates muon trajectories."""
import sys
from copy import copy
from itertools import product

import numpy as np

from muonsim import utils

deg = np.pi / 180.0

class Generator:
    def __init__(
        self, detector, connection, phi_range=None, theta_range=None, walking_hits=False
    ):
        self.detector = detector
        self.connection = connection
        self.phi_range = phi_range
        self.theta_range = theta_range
        self.walking_hits = walking_hits

        self._muons = []
        self._cos_theta_range = None

        self._initialise()

    def _initialise(self):
        """Sanity checks and further initialisations."""

        _theta_range = np.array(self.theta_range)

        self._cos_theta_range = np.cos(_theta_range * deg)
        self._cos_theta_range.sort()

        if not self.connection in self.detector.connections:
            err_msg = f"ERROR: connection {self.connection} is not part of the "
            err_msg += "detector geometry!"
            sys.exit(err_msg)

    def get_muons(self):
        """Returning the generated muons list."""
        return self._muons

    def _walking_step(self, ranges):
        """Generator over 3D points of a region-of-interest."""
        steps = product(
            ranges["top"]["x"],
            ranges["top"]["y"],
            ranges["top"]["z"],
            ranges["bottom"]["x"],
            ranges["bottom"]["y"],
            ranges["bottom"]["z"],
        )

        for t_x, t_y, t_z, b_x, b_y, b_z in steps:
            yield (t_x, t_y, t_z), (b_x, b_y, b_z)

    def _get_roi_segmentation(self, n_divisions):
        """Returns the segmentation of the region-of-interest when generating
        'walking muons'."""

        ranges = {k: None for k in ["x", "y", "z"]}
        panels = {"top": copy(ranges), "bottom": copy(ranges)}
        divisions = {"x": n_divisions, "y": n_divisions, "z": 2}

        n_panel_points = divisions["x"] * divisions["y"] * divisions["z"]
        n_muons = n_panel_points**2

        for p in panels:
            for d in ranges:

                r = self.detector.get_intersection_extension(self.connection, p, d)
                
                panels[p][d] = np.linspace(*r, divisions[d])
        
        return panels, n_muons

    def generate_muons(self, n_samples):
        """Filling the muons dictionary with n_muons generated particles."""
        n_muons = None
        walking_step = None

        if self.walking_hits:
            panels_ranges, n_muons = self._get_roi_segmentation(n_samples)
            walking_step = self._walking_step(panels_ranges)

        else:
            n_muons = n_samples

        print(f"Generating {n_muons} muons for connection {self.connection} ...")

        for m_idx in range(n_muons):
            muon = {"start": None, "stop": None, "theta": None, "phi": None}

            _start, _stop = None, None

            if self.walking_hits:
                # Generating muons on a uniform grid.
                _start, _stop = next(walking_step)

            else:
                # Standard radomised hit generation.
                _start = self.detector.generate_hit(self.connection, "top")
                _stop = self.detector.generate_hit(self.connection, "bottom")

            start = np.array(_start)
            stop = np.array(_stop)

            theta, phi = utils.get_polar_coor(start, stop)

            if self.phi_range:
                phi = np.random.uniform(*self.phi_range)

            if self.theta_range:
                # Generating a uniform distribution requires randomising
                # on cos_theta.
                cos_theta = np.random.uniform(*self._cos_theta_range)
                theta = np.arccos(cos_theta) / deg

            # IMPORTANT: the stop coordinate is never really used.
            # start and stop are both used to trace the muon and obtain
            # its angular coordinates only inside this class.
            muon["start"] = start
            muon["stop"] = stop

            muon["phi"] = phi
            muon["theta"] = theta

            self._muons.append(muon)


# EOF
