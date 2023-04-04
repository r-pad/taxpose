# import cv2
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from equivariant_pose_graph.utils import to_np


def add_color(pts, color):
    c = np.repeat([color], len(pts), axis=0)
    return np.concatenate([pts, c], axis=1)


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def scatter3d(xs, cs=None, figure_size=[9.5, 6], dpi=100):
    figure = Figure(figsize=figure_size, dpi=dpi)

    ax = figure.add_subplot(111, projection="3d")
    for x, c in zip(xs, cs):
        ax.scatter3D(
            x[:, 0].cpu().detach().numpy(),
            x[:, 1].cpu().detach().numpy(),
            x[:, 2].cpu().detach().numpy(),
            c=c,
        )
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    set_axes_equal(ax)

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    img = np.array(canvas.buffer_rgba())
    plt.close(figure)
    return img


def quiver3d(xs, vs, cxs=None, cvs=None, s=0.5, figure_size=[9.5, 6], dpi=100):
    figure = Figure(figsize=figure_size, dpi=dpi)

    ax = figure.add_subplot(111, projection="3d")
    for x, v, cx, cv in zip(xs, vs, cxs, cvs):
        if cx is not None:
            ax.scatter3D(
                x[:, 0].cpu().detach().numpy(),
                x[:, 1].cpu().detach().numpy(),
                x[:, 2].cpu().detach().numpy(),
                c=cx,
                s=s,
            )
        if cv is not None:
            ax.quiver(
                x[:, 0].cpu().detach().numpy(),
                x[:, 1].cpu().detach().numpy(),
                x[:, 2].cpu().detach().numpy(),
                v[:, 0].cpu().detach().numpy(),
                v[:, 1].cpu().detach().numpy(),
                v[:, 2].cpu().detach().numpy(),
                color=cv,
                linewidth=0.5,
                length=1.0,
                normalize=False,
            )

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    set_axes_equal(ax)

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    img = np.array(canvas.buffer_rgba())
    plt.close(figure)
    return img
