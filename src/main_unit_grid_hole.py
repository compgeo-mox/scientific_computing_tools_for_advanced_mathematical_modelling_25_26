"""Run elasticity on the unit-square grid with hole."""

from pathlib import Path

from create_grid import create_grid_with_hole
from common_functions import solve_and_export

folder = Path(__file__).parent


def main():
    """Run the unit-grid-with-hole elasticity workflow.

    Parameters
    ----------
    None
        The script uses fixed settings defined inside the function.

    Returns
    -------
    None
        Solution files are written to the results directory.
    """
    mesh_size = 0.1

    # Step 1: generate the unit grid with noisy elliptical hole.
    sd = create_grid_with_hole(mesh_size)

    # Step 2: solve elasticity and export displacement.
    folder_export = folder / "results"
    solve_and_export(sd, folder_export, export_name="sol_unit_hole")


if __name__ == "__main__":
    main()
