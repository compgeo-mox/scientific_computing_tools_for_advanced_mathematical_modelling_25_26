"""Utilities to assemble, solve, and export a linear elasticity problem."""

from pathlib import Path

import numpy as np

import pygeon as pg
import porepy as pp


class ElasticProblem:
    """Wrapper for the linear-elasticity workflow on one grid.

    Parameters
    ----------
    sd : pg.Grid
        Spatial discretization used by assembly and post-processing methods.
    key : str, optional
        Data-dictionary key under which elasticity parameters are stored.

    Returns
    -------
    ElasticProblem
        Instance exposing assembly, solve, and export utilities.
    """

    def __init__(self, sd, key="elasticity"):
        """Initialize the elasticity helper.

        Parameters
        ----------
        sd : pg.Grid
            Spatial discretization used by assembly and post-processing methods.
        key : str, optional
            Data-dictionary key under which elasticity parameters are stored.

        Returns
        -------
        None
            Constructor initializes object state in place.
        """
        self.sd = sd
        self.key = key

        self.vec_p1 = pg.VecLagrange1(self.key)

    def assemble_problem(self, param, body_force, nat_bc_faces):
        """Assemble system matrix and natural-boundary right-hand side.

        Parameters
        ----------
        param : dict
            Material and boundary-condition parameters passed to ``pp.initialize_data``.
        body_force : callable
            Callback used by ``assemble_nat_bc`` to evaluate boundary traction.
        nat_bc_faces : np.ndarray
            Boolean mask or index array selecting faces with natural boundary
            conditions.

        Returns
        -------
        tuple
            ``(A, b)`` where ``A`` is the assembled stiffness matrix and ``b``
            is the natural-boundary contribution to the right-hand side.
        """
        self.data = pp.initialize_data({}, self.key, param)

        # Step 1: assemble the stiffness matrix from material parameters.
        A = self.vec_p1.assemble_stiff_matrix_elasticity(self.sd, self.data)

        # Step 2: assemble Neumann (natural) boundary contribution.
        b = self.vec_p1.assemble_nat_bc(self.sd, body_force, nat_bc_faces)

        return A, b

    def assemble_problem_pressure(self, param, body_force, pressure, nat_bc_faces):
        """Assemble system matrix and natural-boundary right-hand side.

        Parameters
        ----------
        param : dict
            Material and boundary-condition parameters passed to ``pp.initialize_data``.
        body_force : callable
            Callback used by ``assemble_nat_bc`` to evaluate boundary traction.
        nat_bc_faces : np.ndarray
            Boolean mask or index array selecting faces with natural boundary
            conditions.

        Returns
        -------
        tuple
            ``(A, b)`` where ``A`` is the assembled stiffness matrix and ``b``
            is the natural-boundary contribution to the right-hand side.
        """
        self.data = pp.initialize_data({}, self.key, param)

        # Step 1: assemble the stiffness matrix from material parameters.
        A = self.vec_p1.assemble_stiff_matrix_elasticity(self.sd, self.data)

        # Step 2: assemble Neumann (natural) boundary contribution.
        b = self.vec_p1.assemble_nat_bc(self.sd, body_force, nat_bc_faces)

        # Step 3: assemble pressure term
        M = self.vec_p1.assemble_div_matrix(self.sd)
        b += M.T@pressure
        return A, b
    
    def solve_linear_system(self, A, b, ess_bc_faces):
        """Solve the linear system with essential (Dirichlet) constraints.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Assembled linear-system matrix.
        b : np.ndarray
            Linear-system right-hand side.
        ess_bc_faces : np.ndarray
            Boolean mask or index array selecting displacement dofs constrained
            by essential boundary conditions.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(u, sigma)`` where ``u`` is the displacement field (padded to
            three components in 2-D for VTK output) and ``sigma`` is the
            computed stress tensor per cell.
        """
        # Step 1: build the linear system and enforce essential BCs.
        ls = pg.LinearSystem(A, b)
        ls.flag_ess_bc(ess_bc_faces, np.zeros(self.vec_p1.ndof(self.sd)))

        # Step 2: solve for nodal displacement unknowns.
        u = ls.solve()

        sigma = self.vec_p1.compute_stress(self.sd, u, self.data)

        # Step 3: in 2-D, pad with a zero z-component for visualization tools.
        if self.sd.dim == 2:
            u = np.hstack((u, np.zeros(self.sd.num_nodes))).reshape((3, -1))

        return u, sigma

    def export_solution(
        self, u, sigma, folder_export, export_name="sol", cell_data=None
    ):
        """Export displacement and stress data to VTU/PVD files.

        Parameters
        ----------
        u : np.ndarray
            Displacement field to export as point data.
        sigma : np.ndarray
            Stress tensor field used to export scalar cell components.
        folder_export : str or pathlib.Path
            Output directory where export files are written.
        export_name : str, optional
            Base filename used by the exporter.
        cell_data : list[tuple[str, np.ndarray]] or None, optional
            Optional additional cell fields appended to default stress outputs.

        Returns
        -------
        None
            Files are written to disk; the method does not return a value.
        """
        # Create exporter and write point data + stress components as cell data.
        export_folder = Path(folder_export)
        save = pp.Exporter(self.sd, export_name, folder_name=export_folder)

        # Export scalar stress components per cell for easier post-processing.
        data = [
            ("cell_sigma_xx", sigma[0, 0, :]),
            ("cell_sigma_xy", sigma[0, 1, :]),
            ("cell_sigma_yy", sigma[1, 1, :]),
        ]

        # Append any caller-provided cell fields (for example layer id).
        if cell_data:
            data.extend(cell_data)

        save.write_vtu(data_pt=[("u", u)], data=data)
