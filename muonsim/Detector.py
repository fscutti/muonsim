"""This module builds the muon detector and its plotter."""
import math
import sys

import pyvista as pv
import numpy as np

from itertools import product


class Detector:
    def __init__(self, elements):
        self.elements = elements

        # The _elements attribute represents the active pyvista volumes
        # of the detector. The detector is inserted in a volume which
        # will be used to draw a muon ray. It is computed using the maximal
        # detector dimensions.
        self._elements = {}
        self._elements_labels = {}

        _max_x, _max_y, _max_z = 0, 0, 0
        _min_x, _min_y, _min_z = 1e9, 1e9, 1e9

        for e, coord in self.elements.items():
            self._elements[e] = pv.Box(bounds=(*coord["x"], *coord["y"], *coord["z"]))
            self._elements_labels[e] = coord["center"]

            _max_x, _min_x = max(_max_x, *coord["x"]), min(_min_x, *coord["x"])
            _max_y, _min_y = max(_max_y, *coord["y"]), min(_min_y, *coord["y"])
            _max_z, _min_z = max(_max_z, *coord["z"]), min(_min_z, *coord["z"])

        self.volume = {
            "x": [_max_x, _min_x],
            "y": [_max_y, _min_y],
            "z": [_max_z, _min_z],
        }

        # This is used for drawing the volume of the detector.
        self.pv_volume = pv.Box(
            bounds=(*self.volume["x"], *self.volume["y"], *self.volume["z"])
        )

        self._labels = pv.PolyData([ec for e, ec in self._elements_labels.items()])
        self._labels["Detector labels"] = [e for e in self._elements_labels]

        # The class can hold a bunch of muons (different events) at a time.
        self._muons = []

        # The class only holds muon track intersections with the active
        # volume of the detector for one muon at a time.
        # * _event_points will be used for the analysis.
        # * _event_intersections for eventual pyvista plotting.
        self._event_points = None
        self._event_intersections = None

    def _plane_intersection(
        self, plane_normal, plane_point, muon_direction, muon_point
    ):
        """Computes the intersection b/w a muon and a plane."""
        ndotu = plane_normal.dot(muon_direction)

        if abs(ndotu) < 10e-6:
            # No intersection or line is within plane.
            return

        w = muon_point - plane_point
        si = -plane_normal.dot(w) / ndotu

        return w + si * muon_direction + plane_point

    def _versor(self, theta, phi):
        """Returns a versor corresponding to a set of polar coordinates.
        Input units are expected to be in degrees."""
        # theta = 90.0 - elevation
        theta = theta * np.pi / 180.0
        phi = phi * np.pi / 180.0

        x = math.sin(theta) * math.sin(phi)
        y = math.sin(theta) * math.cos(phi)
        z = math.cos(theta)

        v = np.array([x, y, z])

        return v / np.linalg.norm(v)

    def _find_muon_endpoints(self, muon_theta, muon_phi, muon_origin):
        """Finds the intersection between a muon and the planes
        defining the boundary volume of the detector."""

        # These are the muon intersections.
        muon_endpoints = []
        # This is the number of intersections with the boundary planes.
        muon_coincidences = 0

        dimensions = np.array(["x", "y", "z"])
        normal = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        muon_versor = self._versor(muon_theta, muon_phi)

        # IMPORTANT: remember to change the z_bound to make it
        # compatible with the innermost side of the sensitive
        # area.
        for z_bound in self.volume["z"]:
            # The plane normal and point are assumed to be
            # traced from the origin of the system.
            plane_normal = normal["z"]
            plane_point = z_bound * normal["z"]

            # Perform intersection here.
            muon_point = self._plane_intersection(
                plane_normal,
                plane_point,
                muon_versor,
                muon_origin,
            )

            if muon_point is not None:
                muon_x, muon_y, muon_z = muon_point

                is_within_x = min(self.volume["x"]) <= muon_x <= max(self.volume["x"])
                is_within_y = min(self.volume["y"]) <= muon_y <= max(self.volume["y"])

                # Crossings represent a valid intersection with respect to the
                # sensitive area of the detector The algorithm needs to find
                # at least one of these points.
                if is_within_x and is_within_y:
                    muon_coincidences += 1

                # This list includes intersections outside the sensitive area
                # of the detector. We might still be interested in those but
                # the muon has to have at least one crossing.
                muon_endpoints.append(muon_point)

        if len(muon_endpoints) != 2:
            err_msg = f"ERROR: muon endpoints are {len(muon_endpoints)} for \n"
            err_msg += f" theta: {muon_theta} \n "
            err_msg += f" phi: {muon_phi} \n "
            err_msg += f" origin: {muon_origin} \n "
            err_msg += f" Check volume {self.volume} for elements {self.elements}.\n"
            sys.exit(err_msg)

        return muon_endpoints, muon_coincidences

    def intersect(
        self, muon_theta, muon_phi, muon_origin, coincidences=[2], event_modules=[]
    ):
        """Performs the intersection between a muon and the active
        volume of the detector."""

        self._event_points, self._event_intersections = {}, {}

        muon_endpoints, muon_coincidences = self._find_muon_endpoints(
            muon_theta, muon_phi, muon_origin
        )

        # If the muon satisfies the coincidence criterion and the modules intersection
        # it is added to the cache. Its endpoints will be used for raytracing.
        if muon_endpoints and (muon_coincidences in coincidences):
            muon_start, muon_stop = muon_endpoints

            points, intersections = self._element_intersect(muon_start, muon_stop)

            if event_modules:
                if not set(event_modules) == set(points):
                    return

            self._muons.append(pv.Line(muon_start, muon_stop))
            self._event_points, self._event_intersections = points, intersections
            # print("Adding muons", event_modules, points, self._muons)

    def _element_intersect(self, muon_start, muon_stop):
        """Performs intersections with detector elements."""

        event_points, event_intersections = {}, {}

        for element_name, element_obj in self._elements.items():
            # Selecting coincidences of specific modules.
            muon_points, muon_ind = element_obj.ray_trace(muon_start, muon_stop)

            if len(muon_points) > 0:
                # These are numpy arrays.
                event_points[element_name] = muon_points
                # These are pyvista data structures.
                event_intersections[element_name] = pv.PolyData(muon_points)

        return event_points, event_intersections

    def get_event_points(self):
        """Get latest event valid intersection points."""
        event_points = {}
        for element_name, element_points in self._event_points.items():
            if not element_points.size == 0:
                event_points[element_name] = element_points

        return event_points

    def clear_muons(self, max_muons):
        """Clears all loaded muons."""
        if len(self._muons) > max_muons:
            self._muons = []

    def plot(
        self, add_volume=True, add_elements=True, add_muons=True, add_intersections=True
    ):
        """Every time this function is called, it plots what explicitly requested."""
        _plot = pv.Plotter(off_screen=False)

        # Adding detector volume.
        if add_volume:
            _plot.add_mesh(
                self.pv_volume,
                show_edges=True,
                opacity=0.1,
                color="gray",
                lighting=True,
                label="Volume",
            )

        # Adding detector elements.
        if add_elements:
            for element_name, element_obj in self._elements.items():
                _plot.add_mesh(
                    element_obj,
                    show_edges=True,
                    opacity=0.3,
                    color="b",
                    lighting=True,
                    label=f"{element_name}",
                )
                # Plotting the elements labels.
                _plot.add_point_labels(
                    self._labels, "Detector labels", point_size=2, font_size=20
                )

        # Adding all muons loaded into memory.
        if add_muons:
            for m in self._muons:
                _plot.add_mesh(m, color="red", line_width=1, label="Muon")

        # Adding intersection points for the latest event.
        if add_intersections:
            for element_name, element_intersection in self._event_intersections.items():
                if not self._event_points[element_name].size == 0:
                    _plot.add_mesh(
                        element_intersection,
                        color="yellow",
                        point_size=8,
                        label="Intersection",
                    )

        _plot.show()
