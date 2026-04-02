"""Utilities to assemble, solve, and export a linear elasticity problem."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

import pygeon as pg
import porepy as pp


class ElasticProblem:
    """Wrapper for the discrete linear-elasticity workflow on one grid.

    Input:
    A spatial discretization object and an optional data key.

    Output:
    Object exposing methods to assemble, solve, and export elasticity solutions.
    """

    def __init__(self, sd: Any, key: str = "elasticity") -> None:
        """Create an elasticity problem helper.

        Input:
        sd (grid-like discretization object), key (string used in data dictionaries).

        Output:
        Initialized ElasticProblem instance storing the grid and FE operator.
        """
        self.sd = sd
        self.key = key

        self.vec_p1 = pg.VecLagrange1(self.key)

    def assemble_problem(
        self,
        param: dict[str, Any],
        body_force: Any,
        nat_bc_faces: Any,
    ) -> tuple[Any, Any]:
        """Assemble stiffness matrix and natural-boundary right-hand side.

        Input:
        param (material/BC parameters), body_force function, nat_bc_faces indices.

        Output:
        (A, b) with system matrix A and right-hand-side vector b.
        """
        data = pp.initialize_data({}, self.key, param)

        # Step 1: assemble the stiffness matrix from material parameters.
        A = self.vec_p1.assemble_stiff_matrix(self.sd, data)

        # Step 2: assemble Neumann (natural) boundary contribution.
        b = self.vec_p1.assemble_nat_bc(self.sd, body_force, nat_bc_faces)

        return A, b

    def solve_linear_system(
        self,
        A: Any,
        b: Any,
        ess_bc_faces: Any,
    ) -> npt.NDArray[np.float64]:
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

        # Step 3: in 2-D, pad with a zero z-component for visualization tools.
        if self.sd.dim == 2:
            return np.hstack((u, np.zeros(self.sd.num_nodes))).reshape((3, -1))
        else:
            return u

    def export_solution(
        self,
        u: npt.NDArray[np.float64],
        folder_export: str | Path,
        export_name: str = "sol",
    ) -> None:
        """Export displacement data to VTU/PVD files.

        Input:
        u displacement array, folder_export output directory, and export_name
        base name used for written files.

        Output:
        Files written on disk for visualization (no value returned).
        """
        # Create exporter and write point-data field named "u".
        export_folder = Path(folder_export)
        save = pp.Exporter(self.sd, cast(Any, export_name), folder_name=export_folder)
        save.write_vtu(data_pt=[("u", u)])
