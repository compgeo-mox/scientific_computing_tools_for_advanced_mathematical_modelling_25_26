import sys
from pathlib import Path

folder = Path(__file__).parent
sys.path.append(str(folder))

from create_grid import create_grid_with_hole
from common_functions import solve_and_export


"""Run elasticity on the unit-square grid with hole."""


def main():
    """Entry point for unit-grid-with-hole elasticity solve.

    Input:
    No runtime input; mesh and hole settings are defined below.

    Output:
    Writes solution files in the results folder.
    """
    mesh_size = 0.1

    # Step 1: generate the unit grid with noisy elliptical hole.
    sd = create_grid_with_hole(mesh_size)

    # Step 2: solve elasticity and export displacement.
    folder_export = folder / "results"
    solve_and_export(sd, folder_export, export_name="sol_unit_hole")


if __name__ == "__main__":
    main()
