"""Run elasticity on the layered/faulted grid with hole."""

from pathlib import Path

from create_grid import create_layered_grid_with_hole
from common_functions import solve_and_export

folder = Path(__file__).parent


def main():
    """Run the layered-grid-with-hole elasticity workflow.

    Parameters
    ----------
    None
        The script uses fixed settings defined inside the function.

    Returns
    -------
    None
        Solution files are written to the results directory.
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
