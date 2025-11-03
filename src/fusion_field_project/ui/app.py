
import streamlit as st
import numpy as np
import plotly.express as px

from fusion_field_project.coils.geometry import Loop, helmholtz_pair
from fusion_field_project.simulation.grid import make_plane_grid
from fusion_field_project.simulation.biot_savart import field_of_loops
from fusion_field_project.simulation.metrics import field_magnitude, uniformity_score
from fusion_field_project.simulation.plasma_model import beta_estimate

# ------------------------------- Presets (inline) ------------------------------
def preset_single_loop(radius=0.2, current=100.0, turns=50):
    return [Loop(center=np.array([0.0, 0.0, 0.0]),
                 axis=np.array([0.0, 1.0, 0.0]),
                 radius=float(radius), current=float(current), turns=int(turns))]

def preset_helmholtz(radius=0.2, current=100.0, turns=50, separation=None):
    if separation is None:
        separation = radius
    return helmholtz_pair(float(radius), float(current), int(turns), float(separation))

def preset_maxwell_pair(radius=0.2, current=100.0, turns=50):
    # Maxwell pair: separation = sqrt(3) * radius for better uniformity
    sep = float(np.sqrt(3.0) * radius)
    return helmholtz_pair(float(radius), float(current), int(turns), sep)

def preset_tokamak_like(R_major=0.35, r_minor=0.08, current=100.0, turns=20, ncoils=24):
    """
    Rough torus: ncoils circular loops around a major radius R_major.
    Each loop axis is tangent to the torus path in the x–z plane.
    """
    loops = []
    for k in range(int(ncoils)):
        phi = 2*np.pi * k / int(ncoils)
        cx, cz = float(R_major)*np.cos(phi), float(R_major)*np.sin(phi)
        axis = np.array([-np.sin(phi), 0.0, np.cos(phi)])  # tangent, already unit length
        loops.append(Loop(center=np.array([cx, 0.0, cz]),
                          axis=axis,
                          radius=float(r_minor),
                          current=float(current),
                          turns=int(turns)))
    return loops

# ------------------------------- UI -------------------------------------------
st.set_page_config(page_title="Fusion Field Project", layout="wide")
st.title("Fusion Field Project — Magnetic Field Explorer")

# Sidebar: presets & custom controls
st.sidebar.header("Coil Preset")
preset = st.sidebar.selectbox(
    "Choose a preset",
    ["Custom", "Single loop", "Helmholtz", "Maxwell pair", "Tokamak-ish"],
)

st.sidebar.header("Base Parameters")
current = st.sidebar.slider("Current (A)", 0, 500, 100, step=10)
radius = st.sidebar.slider("Radius (m)", 0.05, 0.5, 0.20, step=0.01)
turns = st.sidebar.slider("Turns", 1, 200, 50, step=1)
separation = st.sidebar.slider("Separation (m) (for Helmholtz/Custom pair)", 0.0, 1.0, radius, step=0.01)

st.sidebar.header("Grid")
extent = st.sidebar.slider("Extent (m)", 0.1, 1.0, 0.5, step=0.05)
res = st.sidebar.slider("Resolution", 61, 241, 121, step=20)
plane = st.sidebar.selectbox("Slice Plane", ["xz", "xy", "yz"])

st.sidebar.header("Solver")
nseg = st.sidebar.slider("Segments per loop (accuracy)", 50, 600, 200, step=10)

st.sidebar.header("Plasma (toy)")
pressure = st.sidebar.number_input("Plasma Pressure (Pa)", min_value=0.0, value=200.0, step=50.0)

# Tokamak controls (only when selected)
if preset == "Tokamak-ish":
    st.sidebar.subheader("Tokamak-ish parameters")
    R_major = st.sidebar.slider("Major radius R (m)", 0.20, 0.80, 0.35, step=0.01)
    r_minor = st.sidebar.slider("Minor radius r (m)", 0.03, 0.20, 0.08, step=0.005)
    ncoils  = st.sidebar.slider("Toroidal coils", 8, 64, 24, step=1)

# ------------------------------- Build coils ----------------------------------
if preset == "Single loop":
    loops = preset_single_loop(radius=radius, current=current, turns=turns)

elif preset == "Helmholtz":
    loops = preset_helmholtz(radius=radius, current=current, turns=turns, separation=separation)

elif preset == "Maxwell pair":
    loops = preset_maxwell_pair(radius=radius, current=current, turns=turns)

elif preset == "Tokamak-ish":
    loops = preset_tokamak_like(R_major=R_major, r_minor=r_minor, current=current, turns=turns, ncoils=ncoils)

else:  # Custom
    mode = st.sidebar.radio("Custom geometry", ["Single Loop", "Pair (Helmholtz-style)"], horizontal=True)
    if mode == "Single Loop":
        loops = [Loop(center=np.array([0, 0, 0]),
                      axis=np.array([0, 1, 0]),
                      radius=float(radius), current=float(current), turns=int(turns))]
    else:
        loops = helmholtz_pair(float(radius), float(current), int(turns), float(separation))

# ------------------------------- Field solve ----------------------------------
ax, X, Y, pts = make_plane_grid(extent, res, plane=plane)
B = field_of_loops(loops, pts, nseg=nseg)
Bmag = field_magnitude(B).reshape(res, res)

# ------------------------------- Plots ----------------------------------------
fig = px.imshow(
    Bmag.T, origin="lower",
    labels=dict(color="|B| [T]"),
    x=ax, y=ax,
    title="Magnetic Field Magnitude |B| (Tesla)"
)
st.plotly_chart(fig, use_container_width=True)

# Direction sketch (quasi-quiver)
if plane == "xz":
    Bx = B[:, 0].reshape(res, res); By = B[:, 2].reshape(res, res)
elif plane == "xy":
    Bx = B[:, 0].reshape(res, res); By = B[:, 1].reshape(res, res)
else:
    Bx = B[:, 1].reshape(res, res); By = B[:, 2].reshape(res, res)

step = max(1, res // 25)
xs = X[::step, ::step]; ys = Y[::step, ::step]
bx = Bx[::step, ::step]; by = By[::step, ::step]

quiver = px.scatter(x=xs.ravel(), y=ys.ravel(), title="Field Direction (qualitative)")
quiver.update_traces(marker=dict(size=1))
scale = (ax[-1] - ax[0]) / 20
lines = []
for (x0, y0, ux, uy) in zip(xs.ravel(), ys.ravel(), bx.ravel(), by.ravel()):
    x1 = x0 + scale * ux
    y1 = y0 + scale * uy
    lines.append(dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1))
quiver.update_layout(shapes=lines)
st.plotly_chart(quiver, use_container_width=True)

# ------------------------------- Metrics --------------------------------------
uni = uniformity_score(Bmag[Bmag > 0])
center_B = float(Bmag[res // 2, res // 2])
beta = beta_estimate(pressure, max(center_B, 1e-9))
st.markdown(f"**Center |B|:** {center_B:.4e} T  |  **Uniformity score:** {uni:.3f}  |  **β (toy):** {beta:.3e}")