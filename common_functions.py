import numpy as np

import pygeon as pg

from elastic_pb import ElasticProblem


"""Shared utilities for elasticity solver entry points.

These helpers keep all main scripts consistent in boundary conditions,
material setup, solve workflow, and export behavior.
"""


def _default_traction(_):
    """Return the default constant traction vector.

    Parameters
    ----------
    _ : np.ndarray
        Unused input accepted for compatibility with callback signatures.

    Returns
    -------
    np.ndarray
        Constant traction vector ``[fx, fy]``.
    """
    return np.array([0.0, 0])


def _standard_boundary_masks(sd):
    """Build standard bottom-Dirichlet and top-Neumann masks.

    Parameters
    ----------
    sd : pg.Grid
        Grid object containing node coordinates and face centers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(bottom, top)`` boolean masks, where ``bottom`` selects displacement
        dofs constrained by essential conditions and ``top`` selects faces with
        natural traction conditions.
    """
    # Use geometric min/max so this also works for non-unit domains.
    y_min = float(np.min(sd.nodes[1, :]))
    y_max = float(np.max(sd.face_centers[1, :]))

    # Essential BC acts on vector dofs, so node mask is repeated by sd.dim.
    bottom = np.hstack([np.isclose(sd.nodes[1, :], y_min)] * sd.dim)
    top = np.isclose(sd.face_centers[1, :], y_max)

    return bottom, top

def solve(sd):
    """Solve linear elasticity on a grid

    Parameters
    ----------
    sd : pg.Grid
        Grid used for discretization and assembly.
    folder_export : str or pathlib.Path
        Output directory where exported files are written.
    export_name : str
        Base name used by the exporter for output files.
    cell_data : list[tuple[str, np.ndarray]] or None, optional
        Optional additional cell fields to export.

    Returns
    -------
    np.ndarray
        Displacement field returned by the linear solver.
    """
    # Step 1: define material parameters and standard boundary masks.
    lambda_ = 1.0
    mu = 0.5
    param = {pg.LAME_LAMBDA: lambda_, pg.LAME_MU: mu}
    bottom, top = _standard_boundary_masks(sd)

    # Step 2: assemble and solve the discrete elasticity system.
    elastic_pb = ElasticProblem(sd)
    A, b = elastic_pb.assemble_problem(param, _default_traction, top)
    u, sigma = elastic_pb.solve_linear_system(A, b, bottom)


    return u,sigma

def solve_and_export(sd, folder_export, export_name, cell_data=None):
    """Solve linear elasticity on a grid and export solution fields.

    Parameters
    ----------
    sd : pg.Grid
        Grid used for discretization and assembly.
    folder_export : str or pathlib.Path
        Output directory where exported files are written.
    export_name : str
        Base name used by the exporter for output files.
    cell_data : list[tuple[str, np.ndarray]] or None, optional
        Optional additional cell fields to export.

    Returns
    -------
    np.ndarray
        Displacement field returned by the linear solver.
    """
    # Step 1: define material parameters and standard boundary masks.
    lambda_ = 1.0
    mu = 0.5
    param = {pg.LAME_LAMBDA: lambda_, pg.LAME_MU: mu}
    bottom, top = _standard_boundary_masks(sd)

    # Step 2: assemble and solve the discrete elasticity system.
    elastic_pb = ElasticProblem(sd)
    A, b = elastic_pb.assemble_problem(param, _default_traction, top)
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
def solve_and_export_pressure(sd, layers, layer_ID, pressure_val, folder_export, export_name, cell_data=None):
    """Solve linear elasticity on a grid and export solution fields.

    Parameters
    ----------
    sd : pg.Grid
        Grid used for discretization and assembly.
    folder_export : str or pathlib.Path
        Output directory where exported files are written.
    export_name : str
        Base name used by the exporter for output files.
    cell_data : list[tuple[str, np.ndarray]] or None, optional
        Optional additional cell fields to export.

    Returns
    -------
    np.ndarray
        Displacement field returned by the linear solver.
    """
    # Step 1: define material parameters and standard boundary masks.
    lambda_ = 1.0
    mu = 0.5
    param = {pg.LAME_LAMBDA: lambda_, pg.LAME_MU: mu}
    bottom, top = _standard_boundary_masks(sd)
    sd.compute_geometry()

    pressure = np.array(pressure_val*(layers==layer_ID))

    # Step 2: assemble and solve the discrete elasticity system.
    elastic_pb = ElasticProblem(sd)
    A, b = elastic_pb.assemble_problem_pressure(param, _default_traction,pressure, top)
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

def solve_and_export2(sd, folder_export, export_name, cell_data=None):
    """Solve linear elasticity on a grid and export solution fields.

    Parameters
    ----------
    sd : pg.Grid
        Grid used for discretization and assembly.
    folder_export : str or pathlib.Path
        Output directory where exported files are written.
    export_name : str
        Base name used by the exporter for output files.
    cell_data : list[tuple[str, np.ndarray]] or None, optional
        Optional additional cell fields to export.

    Returns
    -------
    np.ndarray
        Displacement field returned by the linear solver.
    """
    # Step 1: define material parameters and standard boundary masks.
    lambda_ = 1.0
    mu = 0.5
    param = {pg.LAME_LAMBDA: lambda_, pg.LAME_MU: mu}
    bottom, top = _standard_boundary_masks(sd)

    # Step 2: assemble and solve the discrete elasticity system.
    elastic_pb = ElasticProblem(sd)
    A, b = elastic_pb.assemble_problem(param, _default_traction, top)
    u, sigma = elastic_pb.solve_linear_system(A, b, bottom)
    e=np.zeros([3,sigma.shape[2]])
    v=np.zeros([3,3,sigma.shape[2]])
    for i in range(sigma.shape[2]):
        sxx=sigma[0,0,i]
        sxy=sigma[0,1,i]
        syy=sigma[1,1,i]
        S = sigma[0:2,0:2,i]
        e[0:2,i],v[0:2,0:2,i]=np.linalg.eig(S)
    
    # Step 3: export through ElasticProblem only (single exporter path).
    elastic_pb.export_solution(
        u,
        sigma,
        folder_export,
        export_name=export_name,
        cell_data=cell_data,
    )
    elastic_pb.export_solution(
        u,
        v,
        folder_export,
        export_name="sforzi_princ",
        cell_data=cell_data,
    )
    return u,sigma