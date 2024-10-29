"""This module builds the muon detector and its plotter."""
import sys

import pyvista as pv
import numpy as np

from itertools import product
from itertools import chain

from muonsim import utils


class Detector:
    def __init__(self, elements, connections, areas):
        self.elements = elements
        self.connections = connections
        self.areas = areas

        # The _elements attribute represents the active pyvista volumes
        # of the detector. The detector is inserted in a volume which
        # will be used to draw a muon ray. It is computed using the maximal
        # detector dimensions.
        self._elements = {}
        self._elements_labels = {}

        self._elements_connections = {}
        self._elements_areas = {}

        for c, extrema in self.connections.items():
            self._elements_connections[c] = pv.Line(*extrema)

            top_area, bottom_area = self.areas[c]
            top_bounds = list(chain.from_iterable(top_area))
            bottom_bounds = list(chain.from_iterable(bottom_area))

            self._elements_areas[c] = [
                pv.Box(bounds=top_bounds),
                pv.Box(bounds=bottom_bounds),
            ]

        _max_x, _max_y, _max_z = 0, 0, 0
        _min_x, _min_y, _min_z = 1e9, 1e9, 1e9

        for e, coord in self.elements.items():
            self._elements[e] = pv.Box(bounds=(*coord["x"], *coord["y"], *coord["z"]))
            self._elements_labels[e] = coord["center"]

            _max_x, _min_x = max(_max_x, *coord["x"]), min(_min_x, *coord["x"])
            _max_y, _min_y = max(_max_y, *coord["y"]), min(_min_y, *coord["y"])
            _max_z, _min_z = max(_max_z, *coord["z"]), min(_min_z, *coord["z"])

        # This expansion factor is needed to make sure that all elements are
        # comfortably contained within the detector volume such that the
        # element intersections with the generated muons, at the edge of the
        # volume, are meaningful. It should not be too large or too small.
        _expansion_factor = 1.001

        self.volume = {
            "x": [_expansion_factor * _max_x, _expansion_factor * _min_x],
            "y": [_expansion_factor * _max_y, _expansion_factor * _min_y],
            "z": [_expansion_factor * _max_z, _expansion_factor * _min_z],
        }

        # This is used for drawing the volume of the detector.
        self.pv_volume = pv.Box(
            bounds=(*self.volume["x"], *self.volume["y"], *self.volume["z"])
        )

        self._labels = pv.PolyData(list(self._elements_labels.values()))
        self._labels["Detector labels"] = list(self._elements_labels.keys())

        # The class can hold a bunch of muons (different events) at a time.
        self._muons = []

        # This is the current event. These should be lists of numpy arrays.
        self._true_muon = None
        self._reconstructed_muon = None

        # The class only holds muon track intersections with the active
        # volume of the detector for one muon at a time.
        # * _muon_hits will be used for the analysis.
        # * _muon_intersections for eventual pyvista plotting.
        self._muon_hits = None
        self._muon_intersections = None

        # This dictionary is used to collect all those elements intersections
        # which have a number of intersections different than two. This should
        # never happen and in these cases we are interested in plotting these
        # bad events.
        self._bad_muon_hits = []
        self._bad_muon_intersections = []

    def get_intersection_extension(self, connection, panel, coord):
        """Returns the boundaries of the area traversed at
        the center by a connection."""

        p_idx = ["top", "bottom"].index(panel)

        if coord == "x":
            return [
                self._elements_areas[connection][p_idx].bounds[0],
                self._elements_areas[connection][p_idx].bounds[1],
            ]

        elif coord == "y":
            return [
                self._elements_areas[connection][p_idx].bounds[2],
                self._elements_areas[connection][p_idx].bounds[3],
            ]

        elif coord == "z":
            return [
                self._elements_areas[connection][p_idx].bounds[4],
                self._elements_areas[connection][p_idx].bounds[5],
            ]

        else:
            err_msg = f"ERROR: trying to get {coord} extension"
            err_msg += " for connection {connetion} and {panel} panel"
            err_msg += " but the area is not found by the Detector class."
            sys.exit(err_msg)

    def get_intersection_center(self, connection, panel, coord):
        """Returns the central coordinates of the area traversed
        by a connection."""

        p_idx = ["top", "bottom"].index(panel)

        if coord == "x":
            return (
                self._elements_areas[connection][p_idx].bounds[0]
                + self._elements_areas[connection][p_idx].bounds[1]
            ) / 2.0

        elif coord == "y":
            return (
                self._elements_areas[connection][p_idx].bounds[2]
                + self._elements_areas[connection][p_idx].bounds[3]
            ) / 2.0

        elif coord == "z":
            return (
                self._elements_areas[connection][p_idx].bounds[4]
                + self._elements_areas[connection][p_idx].bounds[5]
            ) / 2.0

        else:
            err_msg = f"ERROR: trying to get {coord} center"
            err_msg += " for connection {connetion} and {panel} panel"
            err_msg += " but the area is not found by the Detector class."
            sys.exit(err_msg)

    def _find_muon_endpoints(self, muon_theta, muon_phi, muon_origin):
        """Finds the intersection between a muon and the planes
        defining the boundary volume of the detector."""

        # These are the muon intersections.
        muon_endpoints = []
        # This is the number of intersections with the boundary planes.
        muon_boundary_coincidences = 0

        normal = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        muon_versor = utils.get_versor(muon_theta, muon_phi)

        # Notice that the z_bound is chosen compatibly with the
        # entire active volume of the detector. This active volume
        # should not used in the generation of the muon hits but only
        # to search for muon intersections.
        for z_bound in self.volume["z"]:
            # The plane normal and point are assumed to be
            # traced from the origin of the system.
            plane_normal = normal["z"]
            plane_point = z_bound * normal["z"]

            # Perform intersection here. muon_hit is a numpy array.
            muon_hit = utils.get_plane_intersection(
                plane_normal,
                plane_point,
                muon_versor,
                muon_origin,
            )

            if muon_hit is not None:
                muon_x, muon_y, muon_z = muon_hit

                is_within_x = min(self.volume["x"]) <= muon_x <= max(self.volume["x"])
                is_within_y = min(self.volume["y"]) <= muon_y <= max(self.volume["y"])

                # Crossings represent a valid intersection with respect to the
                # sensitive area of the detector The algorithm needs to find
                # at least one of these points.
                if is_within_x and is_within_y:
                    muon_boundary_coincidences += 1

                # This list includes intersections outside the sensitive area
                # of the detector. We might still be interested in those but
                # the muon has to have at least one crossing.
                muon_endpoints.append(muon_hit)

        if len(muon_endpoints) != 2:
            err_msg = f"ERROR: muon endpoints are {len(muon_endpoints)} for \n"
            err_msg += f" theta: {muon_theta} \n "
            err_msg += f" phi: {muon_phi} \n "
            err_msg += f" origin: {muon_origin} \n "
            err_msg += f" Check volume {self.volume} for elements {self.elements}.\n"
            sys.exit(err_msg)

        return muon_endpoints, muon_boundary_coincidences

    def reset_event(self):
        """Resets the event information. Notice that the bad hits are never reset."""
        self._muon_hits, self._muon_intersections = {}, {}
        self._true_muon, self._reconstructed_muon = None, None

    def intersect(
        self,
        muon_theta,
        muon_phi,
        muon_origin,
        required_boundary_coincidences,
        required_modules,
    ):
        """Performs the intersection between a muon and the active
        volume of the detector."""

        muon_endpoints, muon_boundary_coincidences = self._find_muon_endpoints(
            muon_theta, muon_phi, muon_origin
        )

        # If the muon satisfies the coincidence criterion and the modules intersection
        # it is added to the cache. Its endpoints will be used for raytracing.
        if muon_endpoints and (
            muon_boundary_coincidences in required_boundary_coincidences
        ):
            muon_start, muon_stop = muon_endpoints

            (
                hits,
                intersections,
                bad_hits,
                bad_intersections,
            ) = self._element_intersect(muon_start, muon_stop)

            if required_modules:
                if not set(required_modules) == set(hits):
                    return False

            self._muons.append([muon_start, muon_stop])

            self._muon_hits, self._muon_intersections = hits, intersections

            if bad_hits:
                self._bad_muon_hits.append(bad_hits)
                self._bad_muon_intersections.append(bad_intersections)

            self._true_muon = [muon_start, muon_stop]

            return True

        return False

    def _element_intersect(self, muon_start, muon_stop):
        """Performs intersections with detector elements."""

        muon_hits, muon_intersections = {}, {}
        bad_muon_hits, bad_muon_intersections = {}, {}

        # print("----------------------")
        # print(f"Start: {muon_start}")
        # print(f"Stop: {muon_stop}")
        # print("----------------------")

        for element_name, element_obj in self._elements.items():
            # Selecting coincidences of specific modules.
            element_hits, element_ind = element_obj.ray_trace(muon_start, muon_stop)

            if len(element_hits) > 0:
                if len(element_hits) == 2:
                    # These are numpy arrays.
                    muon_hits[element_name] = element_hits
                    # These are pyvista data structures.
                    muon_intersections[element_name] = pv.PolyData(element_hits)

                elif len(element_hits) == 1:
                    bad_muon_hits[element_name] = element_hits
                    bad_muon_intersections[element_name] = pv.PolyData(element_hits)

        if len(muon_hits) == 0:
            # If muon_hits is empty, the muon is within the volume but the volume
            # might be too large for the muon to intersect any active element.
            print("WARNING: muon hits should not be empty at this point.")

        return (
            muon_hits,
            muon_intersections,
            bad_muon_hits,
            bad_muon_intersections,
        )

    def get_muon_hits(self):
        """Get latest event valid intersection hits."""
        # WARNING: many times this dictionary will be empty. These are cases
        # where the generated muon does not satisfy our selection criteria
        # e.g. coincidences. It is up to the main selection to remove these
        # hits.
        return self._muon_hits

    def get_muon_event(self, mode="angles"):
        """Get the true and reconstructed muons, either their angles or endpoints."""
        if mode == "endpoints":
            return self._true_muon, self._reconstructed_muon
        return utils.get_polar_coor(*self._true_muon), utils.get_polar_coor(
            *self._reconstructed_muon
        )

    def generate_hit(self, connection, panel, is_reconstruction=False):
        """Generation of a muon hit in a detector panel. N.B the z coordinate
        is kept at a constant value and not randomised."""
        range_x = self.get_intersection_extension(connection, panel, "x")
        range_y = self.get_intersection_extension(connection, panel, "y")
        range_z = self.get_intersection_extension(connection, panel, "z")

        hit_x = np.random.uniform(*range_x)
        hit_y = np.random.uniform(*range_y)
        hit_z = None

        if is_reconstruction:
            hit_z = np.random.uniform(*range_z)
        else:
            hit_z = self.get_intersection_center(connection, panel, "z")

        return [hit_x, hit_y, hit_z]

    def reconstruct(self):
        """Get reconstructed muon endpoints."""
        current_hit_modules = set(self._muon_hits)

        for c, extrema in self.connections.items():
            reconstructed_combination = set(c.split("_"))

            if reconstructed_combination.issubset(current_hit_modules):
                # start, stop = extrema

                start = self.generate_hit(c, "top", is_reconstruction=True)
                stop = self.generate_hit(c, "bottom", is_reconstruction=True)

                self._reconstructed_muon = [np.array(start), np.array(stop)]
                return True

        return False

    def clear_muons(self, max_muons):
        """Clears all loaded muons."""
        if len(self._muons) > max_muons:
            self._muons = []

    def plot(
        self,
        add_volume=True,
        add_elements=True,
        add_connections=True,
        add_muons=True,
        add_intersections=True,
        start=None,
        stop=None,
    ):
        """Every time this function is called, it plots what explicitly requested."""
        _plot = pv.Plotter(off_screen=False)

        _colors = ["red", "green", "blue"]

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

        if start and stop:
            for c_idx, (ss, st) in enumerate(zip(start, stop)):
                line = pv.Line(ss, st)
                _plot.add_mesh(
                    line, color=_colors[c_idx], line_width=6.0, label="Connections"
                )

        if add_connections:
            # Displaying relevant relations b/w detector components.
            for connection_name, connection_obj in self._elements_connections.items():
                _plot.add_mesh(
                    connection_obj, color="green", line_width=0.5, label=connection_name
                )

        # Adding all muons loaded into memory.
        if add_muons:
            print(f"Detector plot: adding {len(self._muons)} muons")

            for m in self._muons:
                _m = pv.Line(*m)
                _plot.add_mesh(_m, color="red", line_width=1.0, label="Muon")

        # Adding intersection points for the latest event.
        if add_intersections:
            print(f"Detector intersection: adding {len(self._muons)} muons")
            print(f"Available event intersections {self._muon_intersections}")

            for element_name, element_intersection in self._muon_intersections.items():
                print(f"Adding latest event intersection with {element_name}:")

                if not self._muon_hits[element_name].size == 0:
                    print(f"    {element_intersection}")
                    _plot.add_mesh(
                        element_intersection,
                        color="green",
                        point_size=8,
                        label="Intersection",
                    )

            print(
                f"Printing {len(self._bad_muon_intersections)} bad event intersections"
            )
            for bad_intersection in self._bad_muon_intersections:
                # print(f"Adding bad event intersection with {element_name}:")

                for (
                    bad_element_name,
                    bad_element_intersection,
                ) in bad_intersection.items():
                    # print(f"    {bad_element_intersection}")
                    _plot.add_mesh(
                        bad_element_intersection,
                        color="yellow",
                        point_size=8,
                        label=bad_element_name,
                    )

        _plot.show()
