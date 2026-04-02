import numpy as np

import pygeon as pg

from elastic_pb import ElasticProblem


"""Shared utilities for elasticity solver entry points.

These helpers keep all main scripts consistent in boundary conditions,
material setup, solve workflow, and export behavior.
"""


def _default_body_force(_):
    """Return the default constant body force [fx, fy].

    Input:
    Unused point coordinates.

    Output:
    Constant 2-D body-force vector.
    """
    return np.array([0.0, -1e-3])


def _standard_boundary_masks(sd):
    """Build standard bottom Dirichlet and top Neumann masks.

    Input:
    sd grid object with nodes and face centers.

    Output:
    (bottom, top) boolean masks for essential and natural boundaries.
    """
    # Use geometric min/max so this also works for non-unit domains.
    y_min = float(np.min(sd.nodes[1, :]))
    y_max = float(np.max(sd.face_centers[1, :]))

    # Essential BC acts on vector dofs, so node mask is repeated by sd.dim.
    bottom = np.hstack([np.isclose(sd.nodes[1, :], y_min)] * sd.dim)
    top = np.isclose(sd.face_centers[1, :], y_max)

    return bottom, top


def solve_and_export(sd, folder_export, export_name, cell_data=None):
    """Solve linear elasticity on a grid and export fields.

    Input:
    sd grid, folder_export path, export_name base name, optional cell_data list.

    Output:
    Writes VTU/PVD files to disk and returns displacement array u.
    """
    # Step 1: define material parameters and standard boundary masks.
    lambda_ = 1.0
    mu = 0.5
    param = {pg.LAME_LAMBDA: lambda_, pg.LAME_MU: mu}
    bottom, top = _standard_boundary_masks(sd)

    # Step 2: assemble and solve the discrete elasticity system.
    elastic_pb = ElasticProblem(sd)
    A, b = elastic_pb.assemble_problem(param, _default_body_force, top)
    u, sigma = elastic_pb.solve_linear_system(A, b, bottom)

    # Step 3: export through ElasticProblem only (single exporter path).
    elastic_pb.export_solution(
        u,
        sigma,
        folder_export,
        export_name=export_name,
        cell_data=cell_data,
    )

    return u
