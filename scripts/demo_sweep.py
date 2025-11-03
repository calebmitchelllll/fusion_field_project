#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from fusion_field_project.coils.geometry import helmholtz_pair
from fusion_field_project.simulation.grid import make_plane_grid
from fusion_field_project.simulation.biot_savart import field_of_loops
from fusion_field_project.simulation.metrics import field_magnitude, uniformity_score

def sweep_helmholtz(radius=0.2, current=100.0, turns=50, extent=0.3, res=121, plane="xz"):
    seps = np.linspace(0.2*radius, 2.0*radius, 15)
    scores, centerB = [], []
    ax, X, Y, pts = make_plane_grid(extent, res, plane=plane)
    for sep in seps:
        loops = helmholtz_pair(radius, current, turns, sep)
        B = field_of_loops(loops, pts, nseg=200)
        Bmag = field_magnitude(B).reshape(res,res)
        mask = Bmag > 0
        scores.append(uniformity_score(Bmag[mask]))
        centerB.append(Bmag[res//2, res//2])
    return seps, np.array(scores), np.array(centerB)

if __name__ == "__main__":
    seps, score, cB = sweep_helmholtz()
    fig1 = plt.figure()
    plt.plot(seps, score, marker="o"); plt.xlabel("Separation (m)"); plt.ylabel("Uniformity score"); plt.grid(True)
    fig1.savefig("sweep_uniformity.png", dpi=200)

    fig2 = plt.figure()
    plt.plot(seps, cB, marker="o"); plt.xlabel("Separation (m)"); plt.ylabel("Center |B| (T)"); plt.grid(True)
    fig2.savefig("sweep_centerB.png", dpi=200)

    print("Saved: sweep_uniformity.png, sweep_centerB.png")