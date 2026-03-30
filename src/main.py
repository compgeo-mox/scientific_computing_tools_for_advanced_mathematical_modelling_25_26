import sys
import numpy as np
from pathlib import Path

import porepy as pp
import pygeon as pg

folder = Path(__file__).parent
sys.path.append(str(folder))

from create_grid import create_grid_with_hole, create_layered_grid_with_hole, layers
from elastic_pb import ElasticProblem


def main():
    """Solve a 2D linear elasticity problem on the generated constrained grid."""

    def body_force(_: np.ndarray) -> np.ndarray:
        return np.array([0, -1e-3])

    mesh_size = 0.1
    sd = create_grid_with_hole(mesh_size)

    lambda_ = 1
    mu = 0.5
    param = {pg.LAME_LAMBDA: lambda_, pg.LAME_MU: mu}

    # Displacement is a vector field: mark essential/natural boundaries.
    bottom = np.hstack([np.isclose(sd.nodes[1, :], 0)] * sd.dim)
    top = np.isclose(sd.face_centers[1, :], 1)

    elastic_pb = ElasticProblem(sd)

    # Assemble and solve the linear system.
    A, b = elastic_pb.assemble_problem(param, body_force, top)
    u = elastic_pb.solve_linear_system(A, b, bottom)

    # Export solution for visualization.
    folder_export = folder / "results"
    elastic_pb.export_solution(u, folder_export)


def example_layers() -> None:
    """Demo: create a layered geological grid and export the layer-index field.

    Builds a 2×1 rectangular domain with four horizontal geological layers
    displaced by an inclined fault.  The layer-index scalar field is exported
    as cell data so it can be opened in ParaView and coloured by layer.

    The returned ``cell_layer`` array (one integer per cell) can also be used
    to assign different PDE data — e.g. Lamé parameters — to each layer::

        lambda_per_layer = [1.0, 2.0, 1.5, 0.8]
        lam = np.array([lambda_per_layer[k] for k in cell_layer])
    """
    mesh_size = 0.05
    sd, cell_layer = layers(
        mesh_size,
        Lx=2.0,
        Ly=1.0,
        n_layers=4,
        frac_x_bottom=0.35,
        frac_x_top=0.60,
        fault_displacement=0.08,
    )

    n_layer_types = int(cell_layer.max()) + 1
    print(f"Grid: {sd.num_cells} cells, {sd.num_nodes} nodes, {n_layer_types} layers")
    for i in range(n_layer_types):
        print(f"  Layer {i}: {int((cell_layer == i).sum())} cells")

    # Export layer index as a cell scalar for visualisation in ParaView.
    # Open results/layers.pvd and colour by the 'layer' field.
    folder_export = folder / "results"
    save = pp.Exporter(sd, "layers", folder_name=str(folder_export))
    save.write_vtu(data=[("layer", cell_layer.astype(float))])


def example_layers_with_hole() -> None:
    """Demo: layered domain with fault plus hole, exported with layer IDs."""
    mesh_size = 0.05
    sd, cell_layer = create_layered_grid_with_hole(
        mesh_size=mesh_size,
        Lx=2.0,
        Ly=1.0,
        n_layers=4,
        frac_x_bottom=0.35,
        frac_x_top=0.60,
        fault_displacement=0.08,
        roughness=0.02,
        n_seg=5,
        rng_seed=42,
        center=(0.5, 0.5),
        axes=(0.25, 0.14),
        n_hole_points=32,
        hole_noise_level=0.05,
        hole_seed=123,
    )

    n_layer_types = int(cell_layer.max()) + 1
    print(
        "Layered+hole grid: "
        f"{sd.num_cells} cells, {sd.num_nodes} nodes, {n_layer_types} layers"
    )
    for i in range(n_layer_types):
        print(f"  Layer {i}: {int((cell_layer == i).sum())} cells")

    folder_export = folder / "results"
    save = pp.Exporter(sd, "layers_hole", folder_name=str(folder_export))
    save.write_vtu(data=[("layer", cell_layer.astype(float))])


if __name__ == "__main__":
    main()
    # example_layers()
    # example_layers_with_hole()
