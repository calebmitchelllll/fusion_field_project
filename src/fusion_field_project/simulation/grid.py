import numpy as np

def make_plane_grid(extent: float, resolution: int, plane: str = "xz"):
    ax = np.linspace(-extent, extent, resolution)
    X, Y = np.meshgrid(ax, ax, indexing="xy")
    if plane == "xz":
        pts = np.c_[X.ravel(), np.zeros_like(X).ravel(), Y.ravel()]
    elif plane == "xy":
        pts = np.c_[X.ravel(), Y.ravel(), np.zeros_like(X).ravel()]
    else:  # "yz"
        pts = np.c_[np.zeros_like(X).ravel(), X.ravel(), Y.ravel()]
    return ax, X, Y, pts