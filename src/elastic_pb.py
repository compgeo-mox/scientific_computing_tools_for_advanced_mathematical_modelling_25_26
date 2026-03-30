"""Utilities to assemble, solve, and export a linear elasticity problem."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

import pygeon as pg
import porepy as pp


class ElasticProblem:
    """Wrapper for the discrete linear-elasticity workflow on a single grid."""

    def __init__(self, sd: Any, key: str = "elasticity") -> None:
        """Create an elasticity problem helper.

        Parameters
        ----------
        sd
            Spatial discretization object (typically a PorePy grid).
        key
            Data dictionary key used by PorePy/PyGeoN operators.
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

        Parameters
        ----------
        param
            Material and boundary-condition parameters expected by PorePy.
        body_force
            Prescribed force values on natural boundary faces.
        nat_bc_faces
            Indices of faces where natural boundary conditions are applied.

        Returns
        -------
        tuple
            The linear operator `A` and right-hand side vector `b`.
        """
        data = pp.initialize_data({}, self.key, param)

        # Assemble linear system.
        A = self.vec_p1.assemble_stiff_matrix(self.sd, data)

        # Assemble natural boundary contribution.
        b = self.vec_p1.assemble_nat_bc(self.sd, body_force, nat_bc_faces)

        return A, b

    def solve_linear_system(
        self,
        A: Any,
        b: Any,
        ess_bc_faces: Any,
    ) -> npt.NDArray[np.float64]:
        """Solve the assembled linear system with essential constraints.

        Parameters
        ----------
        A
            System matrix returned by the finite-element assembly.
        b
            Right-hand side vector of the linear system.
        ess_bc_faces
            Indices of faces with essential (Dirichlet) boundary conditions.

        Returns
        -------
        numpy.ndarray
            Displacement field; in 2D, it is padded to three components for
            VTK compatibility.
        """
        ls = pg.LinearSystem(A, b)
        ls.flag_ess_bc(ess_bc_faces, np.zeros(self.vec_p1.ndof(self.sd)))
        u = ls.solve()

        # Pad to 3D for VTK output compatibility.
        if self.sd.dim == 2:
            return np.hstack((u, np.zeros(self.sd.num_nodes))).reshape((3, -1))
        else:
            return u

    def export_solution(
        self, u: npt.NDArray[np.float64], folder_export: str | Path
    ) -> None:
        """Export displacement data to VTU/PVD files.

        Parameters
        ----------
        u
            Displacement array to be exported.
        folder_export
            Destination directory where result files are written.
        """
        export_folder = Path(folder_export)
        save = pp.Exporter(self.sd, cast(Any, "sol"), folder_name=export_folder)
        save.write_vtu(data_pt=[("u", u)])
