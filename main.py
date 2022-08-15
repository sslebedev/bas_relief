from typing import Optional

import matplotlib.pyplot as plt
import numpy
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

im: object = plt.imread('in.bmp')
im_h: int = im.shape[0]
im_w: int = im.shape[1]

max_row: int = im_h - 1
max_col: int = im_w - 1

triangles_cont = 2 + 4 * 2 + max_row * max_col * 2
data = numpy.zeros(triangles_cont, dtype=mesh.Mesh.dtype)

DEPTH = 15
DEPTH_OFFSET = 5
UNIT = 255
PIXEL_PER_UNIT = 1.0


def pixel_to_height(pixel: [3]) -> int:
    norm_brightness = (0.0 + pixel[0] + pixel[1] + pixel[2]) / (3 * UNIT)
    return DEPTH_OFFSET + norm_brightness * DEPTH


def index_to_plane(index: int) -> float:
    return index / PIXEL_PER_UNIT


def make_vector(row: int, col: int, height: Optional[float] = None) -> [3]:
    # noinspection PyUnresolvedReferences
    return [index_to_plane(row), index_to_plane(col), height if height is not None else pixel_to_height(im[row][col])]


def make_quad(index: int, vectors_ccw: [4]):
    your_mesh.vectors[index * 2][0] = vectors_ccw[0]
    your_mesh.vectors[index * 2][1] = vectors_ccw[1]
    your_mesh.vectors[index * 2][2] = vectors_ccw[2]
    your_mesh.vectors[index * 2 + 1][0] = vectors_ccw[2]
    your_mesh.vectors[index * 2 + 1][1] = vectors_ccw[3]
    your_mesh.vectors[index * 2 + 1][2] = vectors_ccw[0]


def make_plane_quad(index: int, row: int, col: int,
                    height: Optional[float] = None,
                    row_offset: int = 1, col_offset: int = 1):
    make_quad(index, [make_vector(row, col, height),
                      make_vector(row, col + col_offset, height),
                      make_vector(row + row_offset, col + col_offset, height),
                      make_vector(row + row_offset, col, height)])


i = 0
make_plane_quad(i, 0, 0, height=0, row_offset=max_row, col_offset=max_col)
i += 1
make_quad(i, [make_vector(0, 0, 0),
              make_vector(0, max_col, 0),
              make_vector(0, max_col),
              make_vector(0, 0)])
i += 1
make_quad(i, [make_vector(max_row, 0, 0),
              make_vector(max_row, max_col, 0),
              make_vector(max_row, max_col),
              make_vector(max_row, 0)])
i += 1
make_quad(i, [make_vector(0, 0, 0),
              make_vector(max_row, 0, 0),
              make_vector(max_row, 0),
              make_vector(0, 0)])
i += 1
make_quad(i, [make_vector(0, max_col, 0),
              make_vector(max_row, max_col, 0),
              make_vector(max_row, max_col),
              make_vector(0, max_col)])
i += 1

for row in range(0, max_row - 1):
    for col in range(0, max_col - 1):
        make_plane_quad(i, row, col)
        i += 1

your_mesh = mesh.Mesh(data, remove_empty_areas=False)

def plot_this():
    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # Load the STL files and add the vectors to the plot
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

    # Auto scale to the mesh size
    scale = your_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.show()


def save_this():
    your_mesh.save('out.stl')


save_this()
