import sys
from pathlib import Path

folder = Path(__file__).parent
sys.path.append(str(folder))

from create_grid import create_grid
from common_functions import solve_and_export


"""Run elasticity on the plain unit-square grid."""


def main():
    """Entry point for unit-grid elasticity solve.

    Input:
    No runtime input; mesh settings are defined below.

    Output:
    Writes solution files in the results folder.
    """
    mesh_size = 0.1

    # Step 1: generate the plain unit grid.
    sd = create_grid(mesh_size)

    # Step 2: solve elasticity and export displacement.
    folder_export = folder / "results"
    solve_and_export(sd, folder_export, export_name="sol_unit")


if __name__ == "__main__":
    main()
