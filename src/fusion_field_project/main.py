import numpy as np
import typer
from fusion_field_project.coils.geometry import Loop
from fusion_field_project.simulation.grid import make_plane_grid
from fusion_field_project.simulation.biot_savart import field_of_loops
from fusion_field_project.simulation.metrics import field_magnitude

app = typer.Typer(help="CLI for fusion_field_project")

@app.command()
def slice(current: float = 100.0, radius: float = 0.2, turns: int = 50,
          extent: float = 0.5, res: int = 121, plane: str = "xz"):
    loop = Loop(center=np.array([0,0,0]), axis=np.array([0,1,0]), radius=radius, current=current, turns=turns)
    ax, X, Y, pts = make_plane_grid(extent, res, plane=plane)
    B = field_of_loops([loop], pts)
    Bmag = field_magnitude(B).reshape(res,res)
    print(f"center |B| [T]: {Bmag[res//2, res//2]:.6e}")

if __name__ == "__main__":
    app()