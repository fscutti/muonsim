from copy import copy

import numpy as np
import math


def get_corr_weight(h_corr, muon_theta, muon_phi):
    """Get the correction weight from the correction histogram."""

    if h_corr is None:
        return 1.0

    x_corr_idx = h_corr.GetXaxis().FindBin(muon_theta)
    y_corr_idx = h_corr.GetYaxis().FindBin(muon_phi)

    correction_weight = h_corr.GetBinContent(x_corr_idx, y_corr_idx)

    if correction_weight:
        return correction_weight

    return 0.0


def make_pdf(
    new_hist,
    old_hist,
    normalise_angle=False,
    perform_inversion=False,
    perform_smoothing=False,
):
    """Normalise counts of the histogram by the angular bin width."""

    for glob_idx, el_idx, az_idx in bin_loop(old_hist):
        bin_content = old_hist.GetBinContent(glob_idx)

        if bin_content > 0:
            new_bin_content = bin_content

            if perform_inversion:
                new_bin_content = 1.0 / bin_content

            if normalise_angle:
                new_bin_content /= bin_solid_angle(old_hist, el_idx, az_idx)

            new_hist.SetBinContent(glob_idx, new_bin_content)

    if perform_smoothing:
        new_hist.Smooth()

    new_hist.Scale(1.0 / new_hist.Integral())


def bin_solid_angle(hist, el_idx, az_idx):
    """Normalisation factor taking into account angular areas and energy..."""
    # The solid angle is expressed in rad.
    deg = np.pi / 180.0

    # Assuming that the original angles are in degrees and noticing that
    # usually the input is given in zenith.
    el_high = (90.0 - hist.GetXaxis().GetBinUpEdge(el_idx)) * deg
    el_low = (90.0 - hist.GetXaxis().GetBinLowEdge(el_idx)) * deg

    az_high = hist.GetYaxis().GetBinUpEdge(az_idx) * deg
    az_low = hist.GetYaxis().GetBinLowEdge(az_idx) * deg

    solid_angle = np.fabs(np.sin(el_high) - np.sin(el_low))
    solid_angle *= az_high - az_low

    return solid_angle / deg


def fill_max(hist, elevation, azimuth, value):
    """Replaces the content of the bin if value > current_value.
    At the first iteration the histogram will be filled no matter what."""
    glob_idx = hist.FindBin(elevation, azimuth)
    current_value = hist.GetBinContent(glob_idx)
    if value > current_value:
        hist.SetBinContent(glob_idx, value)


def bin_loop(hist):
    """Generator for looping over 2D histogram bins."""
    n_bins_x, n_bins_y = hist.GetNbinsX(), hist.GetNbinsY()

    for x_idx in range(1, n_bins_x + 1):
        for y_idx in range(1, n_bins_y + 1):
            yield hist.GetBin(x_idx, y_idx), x_idx, y_idx


def get_plane_intersection(plane_normal, plane_point, muon_direction, muon_point):
    """Computes the intersection b/w a muon and a plane."""
    ndotu = plane_normal.dot(muon_direction)

    if abs(ndotu) < 10e-6:
        # No intersection or line is within plane.
        return

    w = muon_point - plane_point
    si = -plane_normal.dot(w) / ndotu

    return w + si * muon_direction + plane_point


def get_versor(theta, phi):
    """Returns a numpy versor corresponding to a set of polar coordinates.
    Input units are expected to be in degrees."""
    theta = np.pi * theta / 180.0
    phi = np.pi * phi / 180.0

    x = math.sin(theta) * math.sin(phi)
    y = math.sin(theta) * math.cos(phi)
    z = math.cos(theta)

    v = np.array([x, y, z])

    return v / np.linalg.norm(v)


def get_polar_coor(start, stop):
    """Returns polar coordinates corresponding to 3D vector in degrees.
    The input are two numpy vectors."""
    x, y, z = np.subtract(start, stop)

    theta = 180.0 * math.acos(z / math.sqrt(x**2 + y**2 + z**2)) / np.pi
    phi = 180.0 * math.atan2(x, y) / np.pi

    if phi < 0.0:
        phi += 360.0

    return theta, phi


def angle_distance(angle1, angle2):
    """Calculates the distance between two angles."""
    # Ensure angles are within the range [0, 360).
    angle1 = angle1 % 360.0
    angle2 = angle2 % 360.0

    # Calculate the absolute difference between the angles.
    diff = abs(angle1 - angle2)

    # Take the shorter distance between the angles (360 - diff if it's greater than 180).
    if diff <= 180.0:
        return diff
    else:
        return 360.0 - diff


def swap_extrema(start, stop):
    """Swaps extrema of a list with two elements."""
    start_x, start_y, start_z = start
    stop_x, stop_y, stop_z = stop

    if stop_z > start_z:
        return [stop, start]

    return [start, stop]


def angular_distance_3d(vector1, vector2):
    """Calculates the angular distance between two vectors in 3D space."""
    # Normalize the vectors.
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)

    # Calculate the dot product between the normalized vectors.
    dot_product = np.dot(vector1, vector2)

    # Ensure the dot product is within the valid range [-1, 1] to avoid numerical issues.
    dot_product = max(min(dot_product, 1.0), -1.0)

    # Calculate the angular distance in radians using the arccosine function.
    angle_rad = math.acos(dot_product)

    # Convert radians to degrees.
    angle_deg = 180.0 * angle_rad / np.pi

    return angle_deg
