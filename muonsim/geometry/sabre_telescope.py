"""This module contains detector geometries."""
import numpy as np
from itertools import product

name = "SABRE"

x_side, y_side, z_side = 60.0, 30.0, 5.0
spacing = 2.0

z_coord = [-z_side - spacing, 0.0, z_side + spacing]
y_coord = [0.0, 0.0, 0.0]
x_coord = [0.0, 0.0, 0.0]

# This will order the tuple a bit better.
sensor_centers = []
for z in z_coord:
    for y in y_coord:
        for x in x_coord:
            sensor_centers.append((x, y, z))

detector = {}
for idx, (x, y, z) in enumerate(sensor_centers):
    if z > 0:
        layer = "T"
    elif z == 0:
        layer = "M"
    else:
        layer = "B"

    detector[f"{layer}"] = {"x": [x - x_side / 2.0, x + x_side / 2.0]}
    detector[f"{layer}"].update({"y": [y - y_side / 2.0, y + y_side / 2.0]})
    detector[f"{layer}"].update({"z": [z - z_side / 2.0, z + z_side / 2.0]})
    detector[f"{layer}"].update({"center": [x, y, z]})


connections = {
    "T_B": [[x_coord[0], y_coord[0], z_coord[0]], [x_coord[2], y_coord[2], z_coord[2]]]
}
connections = {
    "T_M": [[x_coord[0], y_coord[0], z_coord[0]], [x_coord[1], y_coord[1], z_coord[1]]]
}
connections = {
    "M_B": [[x_coord[1], y_coord[1], z_coord[1]], [x_coord[2], y_coord[2], z_coord[2]]]
}

# EOF
