"""This module contains detector geometries."""
import numpy as np
from itertools import product

sensor_side = 2.968
sensor_length = 15.0
sensor_height = 1.27

n_sensors = 5
sensors_spacing = 0.04
layers_gap = 0.2

panels_gap = 10.0

step = sensor_side + sensors_spacing
panel_side = (n_sensors - 1) * step
panel_side = sensor_length

c_low, c_high = -panel_side / 2.0, panel_side / 2.0
coord = np.arange(c_low, c_high + step, step=step)

sensor_names = []
for idx in range(1, 6):
    for panel in ["T", "B"]:
        for layer in ["U", "L"]:
            sensor_names.append(f"{panel}{layer}{idx}")

detector = {}
for sensor_name in sensor_names:
    sensor_coord = {}
    sensor_idx = int(sensor_name[-1])

    is_upper_layer = "U" in sensor_name
    is_top_panel = sensor_name.startswith("T")

    if is_top_panel and is_upper_layer:
        z = panels_gap / 2.0 + sensor_height + layers_gap + sensor_height / 2.0

    elif is_top_panel and not is_upper_layer:
        z = panels_gap / 2.0 + sensor_height / 2.0

    elif not is_top_panel and is_upper_layer:
        z = -(panels_gap / 2.0 + sensor_height / 2.0)

    elif not is_top_panel and not is_upper_layer:
        z = -(panels_gap / 2.0 + sensor_height + layers_gap + sensor_height / 2.0)

    if is_upper_layer:
        x, y = coord[sensor_idx - 1] + sensor_side / 2.0, 0

        sensor_coord["x"] = [x - sensor_side / 2.0, x + sensor_side / 2.0]
        sensor_coord["y"] = [y - sensor_length / 2.0, y + sensor_length / 2.0]

    elif not is_upper_layer:
        x, y = 0, coord[sensor_idx - 1] + sensor_side / 2.0

        sensor_coord["x"] = [x - sensor_length / 2.0, x + sensor_length / 2.0]
        sensor_coord["y"] = [y - sensor_side / 2.0, y + sensor_side / 2.0]

    sensor_coord["z"] = [z - sensor_height / 2.0, z + sensor_height / 2.0]
    sensor_coord["center"] = [x, y, z]

    detector[sensor_name] = sensor_coord

# Top panel.
_top_intersections = {}
for upper_sensor in ["TU1", "TU2", "TU3", "TU4", "TU5"]:
    for lower_sensor in ["TL1", "TL2", "TL3", "TL4", "TL5"]:
        min_x = max(detector[upper_sensor]["x"][0], detector[lower_sensor]["x"][0])
        max_x = min(detector[upper_sensor]["x"][1], detector[lower_sensor]["x"][1])

        min_y = max(detector[upper_sensor]["y"][0], detector[lower_sensor]["y"][0])
        max_y = min(detector[upper_sensor]["y"][1], detector[lower_sensor]["y"][1])

        x = (max_x + min_x) / 2.0
        y = (max_y + min_y) / 2.0
        z = panels_gap / 2.0 + sensor_height + layers_gap / 2.0

        _top_intersections[f"{upper_sensor}_{lower_sensor}"] = [x, y, z]

# Bottom panel.
_bottom_intersections = {}
for upper_sensor in ["BU1", "BU2", "BU3", "BU4", "BU5"]:
    for lower_sensor in ["BL1", "BL2", "BL3", "TL4", "BL5"]:
        min_x = max(detector[upper_sensor]["x"][0], detector[lower_sensor]["x"][0])
        max_x = min(detector[upper_sensor]["x"][1], detector[lower_sensor]["x"][1])

        min_y = max(detector[upper_sensor]["y"][0], detector[lower_sensor]["y"][0])
        max_y = min(detector[upper_sensor]["y"][1], detector[lower_sensor]["y"][1])

        x = (max_x + min_x) / 2.0
        y = (max_y + min_y) / 2.0
        z = -(panels_gap / 2.0 + sensor_height + layers_gap / 2.0)

        _bottom_intersections[f"{upper_sensor}_{lower_sensor}"] = [x, y, z]

# Relevant connections. These are meaninful geometrical relations between
# sensitive components of the detector.
connections = {}
for top_sensors, top_intersection in _top_intersections.items():
    for bottom_sensors, bottom_intersection in _bottom_intersections.items():
        connections[f"{top_sensors}_{bottom_sensors}"] = [
            top_intersection,
            bottom_intersection,
        ]

# EOF
