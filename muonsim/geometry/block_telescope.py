"""This module contains detector geometries."""
import numpy as np
from itertools import product

sensor_side = 3.0
n_sensors = 5
sensors_spacing = 0.04

n_panels = 2
panels_gap = 33.0

# We remove one sensor as we are building the arrays
# of the sensors centres (xyz_coord).
step = sensor_side + sensors_spacing
panel_side = (n_sensors - 1) * step
x_length, y_length = panel_side, panel_side

x_low, x_high = -x_length / 2.0, x_length / 2.0
x_coord = np.arange(x_low, x_high + step, step=step)

y_low, y_high = -y_length / 2.0, y_length / 2.0
y_coord = np.arange(y_low, y_high + step, step=step)

z_low = (panels_gap + sensor_side) / 2.0
z_high = -(panels_gap + sensor_side) / 2.0
z_coord = np.array([z_low, z_high])

# This will order the tuple a bit better.
sensor_centers = []
for z in z_coord:
    for y in y_coord:
        for x in x_coord:
            sensor_centers.append((x, y, z))

detector = {}
for idx, (x, y, z) in enumerate(sensor_centers):
    sensor_id = idx % (n_sensors * n_sensors)
    layer = "T" if z > 0 else "B"

    detector[f"{layer}{sensor_id}"] = {
        "x": [x - sensor_side / 2.0, x + sensor_side / 2.0]
    }
    detector[f"{layer}{sensor_id}"].update(
        {"y": [y - sensor_side / 2.0, y + sensor_side / 2.0]}
    )
    detector[f"{layer}{sensor_id}"].update(
        {"z": [z - sensor_side / 2.0, z + sensor_side / 2.0]}
    )
    detector[f"{layer}{sensor_id}"].update({"center": [x, y, z]})

# EOF
