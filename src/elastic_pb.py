"""Utilities to assemble, solve, and export a linear elasticity problem."""

from pathlib import Path

import numpy as np

import pygeon as pg
import porepy as pp


class ElasticProblem:
    """Wrapper for the discrete linear-elasticity workflow on one grid.

    Input:
    A spatial discretization object and an optional data key.

    Output:
    Object exposing methods to assemble, solve, and export elasticity solutions.
    """

    def __init__(self, sd, key="elasticity"):
        """Create an elasticity problem helper.

        Input:
        sd (grid-like discretization object), key (string used in data dictionaries).

        Output:
        Initialized ElasticProblem instance storing the grid and FE operator.
        """
        self.sd = sd
        self.key = key

        self.vec_p1 = pg.VecLagrange1(self.key)

    def assemble_problem(self, param, body_force, nat_bc_faces):
        """Assemble stiffness matrix and natural-boundary right-hand side.

        Input:
        param (material/BC parameters), body_force function, nat_bc_faces indices.

        Output:
        (A, b) with system matrix A and right-hand-side vector b.
        """
        self.data = pp.initialize_data({}, self.key, param)

        # Step 1: assemble the stiffness matrix from material parameters.
        A = self.vec_p1.assemble_stiff_matrix_elasticity(self.sd, self.data)

        # Step 2: assemble Neumann (natural) boundary contribution.
        b = self.vec_p1.assemble_nat_bc(self.sd, body_force, nat_bc_faces)

        return A, b

    def solve_linear_system(self, A, b, ess_bc_faces):
        """Solve the linear system with essential (Dirichlet) constraints.

        Input:
        A (system matrix), b (right-hand side), ess_bc_faces indices.

        Output:
        Displacement array u; in 2-D it is padded to 3 components for VTK.
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
        """Export displacement data to VTU/PVD files.

        Input:
        u displacement array, sigma stress array, folder_export output directory,
        export_name base name, and optional extra cell_data.

        Output:
        Files written on disk for visualization (no value returned).
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
