import sys
from pathlib import Path

folder = Path(__file__).parent
sys.path.append(str(folder))

from create_grid import create_layered_grid_with_hole
from common_functions import solve_and_export


"""Run elasticity on the layered/faulted grid with hole."""


def main():
    """Entry point for layered-grid-with-hole elasticity solve.

    Input:
    No runtime input; geometric and hole settings are defined below.

    Output:
    Writes solution files in the results folder.
    """
    mesh_size = 0.05

    # Step 1: generate layered/faulted grid with hole and layer id.
    # Use function defaults for geometry, fault, and hole settings.
    sd, cell_layer = create_layered_grid_with_hole(mesh_size=mesh_size)

    # Step 2: solve elasticity and export displacement + layer id.
    folder_export = folder / "results"
    solve_and_export(
        sd,
        folder_export,
        export_name="sol_layered_hole",
        cell_data=[("layer", cell_layer.astype(float))],
    )


if __name__ == "__main__":
    main()
