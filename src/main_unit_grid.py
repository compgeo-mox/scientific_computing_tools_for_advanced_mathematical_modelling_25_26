"""Run elasticity on the plain unit-square grid."""

from pathlib import Path

from create_grid import create_grid
from common_functions import solve_and_export

folder = Path(__file__).parent


def main():
    """Run the unit-grid elasticity workflow.

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

    # Step 1: generate the plain unit grid.
    sd = create_grid(mesh_size)

    # Step 2: solve elasticity and export displacement.
    folder_export = folder / "results"
    solve_and_export(sd, folder_export, export_name="sol_unit")


if __name__ == "__main__":
    main()
