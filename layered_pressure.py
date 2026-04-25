"""Run elasticity on the layered/faulted grid."""

from pathlib import Path

from create_grid import layers
from common_functions import solve_and_export_pressure

folder = Path(__file__).parent


def main():
    """Run the layered-grid elasticity workflow.

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

    # Step 1: generate layered/faulted grid and per-cell layer id.
    # Use function defaults for geometry and fault settings.
    sd, cell_layer = layers(mesh_size)

    # Step 2: solve elasticity and export displacement + layer id.
    #give the desired pressure and the layer ID where you want to apply it
    layer_ID = 2
    pressure_value = 1

    folder_export = folder / "results"
    solve_and_export_pressure(
        sd,
        cell_layer,
        layer_ID,
        pressure_value,
        folder_export,
        export_name="sol_layered",
        cell_data=[("layer", cell_layer.astype(float))],
    )


if __name__ == "__main__":
    main()
