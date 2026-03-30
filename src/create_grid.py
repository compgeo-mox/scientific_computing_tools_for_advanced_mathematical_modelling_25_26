import numpy as np
import numpy.typing as npt

import pygeon as pg
import porepy as pp


ArrayF = npt.NDArray[np.float64]
ArrayI = npt.NDArray[np.int_]


def _polygon_to_3d(vertices_xy: ArrayF) -> ArrayF:
    """Convert a 2xN polygon to PorePy's 3xN format (z = 0)."""
    return np.vstack((vertices_xy, np.zeros((1, vertices_xy.shape[1]))))


def _remove_duplicate_vertices(vertices_xy: ArrayF, tol: float = 1e-12) -> ArrayF:
    """Drop consecutive duplicate vertices to avoid degenerate polygon edges."""
    keep = np.ones(vertices_xy.shape[1], dtype=bool)
    diffs = np.linalg.norm(np.diff(vertices_xy, axis=1), axis=0)
    keep[1:] = diffs > tol
    cleaned = vertices_xy[:, keep]
    if cleaned.shape[1] >= 2 and np.linalg.norm(cleaned[:, 0] - cleaned[:, -1]) <= tol:
        cleaned = cleaned[:, :-1]
    return cleaned


def _points_in_polygon(sd: pg.Grid, polygon_xy: ArrayF) -> npt.NDArray[np.bool_]:
    """Return a boolean mask for cell centers inside the given polygon."""
    poly = _polygon_to_3d(_remove_duplicate_vertices(polygon_xy))
    _, _, in_poly = pp.distances.points_polygon(sd.cell_centers, poly)
    return in_poly


def _mesh_from_domain(
    mesh_size: float,
    Lx: float,
    Ly: float,
    constraints: list[pp.LineFracture],
) -> pg.Grid:
    """Create a 2-D constrained mesh in the physical box [0, Lx] x [0, Ly]."""
    domain = pp.Domain(bounding_box={"xmin": 0.0, "xmax": Lx, "ymin": 0.0, "ymax": Ly})
    sd = pg.grid_from_domain(
        domain,
        mesh_size,
        as_mdg=False,
        fractures=constraints,
        constraints=np.arange(len(constraints)),
    )
    sd.compute_geometry()
    return sd


def _fault_x(
    y: ArrayF | float, Lx: float, Ly: float, xb: float, xt: float
) -> ArrayF | float:
    """Physical x-coordinate of the fault at height y."""
    return (xb + (xt - xb) * (np.asarray(y) / Ly)) * Lx


def _fault_segment(
    y0: float,
    y1: float,
    Lx: float,
    Ly: float,
    frac_x_bottom: float,
    frac_x_top: float,
    n: int = 5,
) -> ArrayF:
    """Polyline on the fault between two y-levels in physical coordinates."""
    y_vals = np.linspace(y0, y1, n)
    x_vals = _fault_x(y_vals, Lx, Ly, frac_x_bottom, frac_x_top)
    return np.vstack((x_vals, y_vals))


def _noisy_polyline(
    x0: float,
    x1: float,
    y_mean: float,
    n_seg: int,
    roughness: float,
    rng: np.random.Generator,
) -> tuple[list[pp.LineFracture], ArrayF]:
    """Create a noisy polyline and its segment constraints."""
    x_pts = np.linspace(x0, x1, n_seg + 1)
    y_pts = np.full(n_seg + 1, y_mean)
    if n_seg > 1:
        y_pts[1:-1] += rng.normal(0.0, roughness, size=n_seg - 1)

    segments: list[pp.LineFracture] = []
    for k in range(n_seg):
        seg = np.array([[x_pts[k], x_pts[k + 1]], [y_pts[k], y_pts[k + 1]]])
        segments.append(pp.LineFracture(seg))

    return segments, np.vstack((x_pts, y_pts))


def _build_layer_constraints(
    Lx: float,
    Ly: float,
    n_layers: int,
    frac_x_bottom: float,
    frac_x_top: float,
    fault_displacement: float,
    roughness: float,
    n_seg: int,
    rng_seed: int,
) -> tuple[list[pp.LineFracture], list[ArrayF], list[ArrayF]]:
    """Build fault + rough layer constraints and store per-layer boundaries."""
    rng = np.random.default_rng(rng_seed)

    constraints: list[pp.LineFracture] = []
    left_bounds: list[ArrayF] = [np.empty((2, 0)) for _ in range(n_layers + 1)]
    right_bounds: list[ArrayF] = [np.empty((2, 0)) for _ in range(n_layers + 1)]

    x_fault_bottom = frac_x_bottom * Lx
    x_fault_top = frac_x_top * Lx
    left_bounds[0] = np.array([[0.0, x_fault_bottom], [0.0, 0.0]])
    left_bounds[n_layers] = np.array([[0.0, x_fault_top], [Ly, Ly]])
    right_bounds[0] = np.array([[x_fault_bottom, Lx], [0.0, 0.0]])
    right_bounds[n_layers] = np.array([[x_fault_top, Lx], [Ly, Ly]])

    # Fault line.
    constraints.append(
        pp.LineFracture(np.array([[x_fault_bottom, x_fault_top], [0.0, Ly]]))
    )

    # Rough interfaces for each layer boundary.
    for i in range(1, n_layers):
        y_left = (i / n_layers) * Ly
        y_right = float(np.clip(i / n_layers + fault_displacement, 0.0, 1.0)) * Ly

        xf_left = float(_fault_x(y_left, Lx, Ly, frac_x_bottom, frac_x_top))
        xf_right = float(_fault_x(y_right, Lx, Ly, frac_x_bottom, frac_x_top))

        left_fracs, left_poly = _noisy_polyline(
            0.0, xf_left, y_left, n_seg, roughness, rng
        )
        right_fracs, right_poly = _noisy_polyline(
            xf_right, Lx, y_right, n_seg, roughness, rng
        )

        constraints.extend(left_fracs)
        constraints.extend(right_fracs)
        left_bounds[i] = left_poly
        right_bounds[i] = right_poly

    return constraints, left_bounds, right_bounds


def _left_piece_polygon(
    lower: ArrayF,
    upper: ArrayF,
    Lx: float,
    Ly: float,
    frac_x_bottom: float,
    frac_x_top: float,
) -> ArrayF:
    """Build closed polygon for one left-block layer piece."""
    fault_up = _fault_segment(
        lower[1, -1], upper[1, -1], Lx, Ly, frac_x_bottom, frac_x_top
    )
    left_down = np.array([[0.0, 0.0], [upper[1, 0], lower[1, 0]]])
    return np.column_stack((lower, fault_up[:, 1:], np.fliplr(upper), left_down[:, 1:]))


def _right_piece_polygon(
    lower: ArrayF,
    upper: ArrayF,
    Lx: float,
    Ly: float,
    frac_x_bottom: float,
    frac_x_top: float,
) -> ArrayF:
    """Build closed polygon for one right-block layer piece."""
    right_up = np.array([[Lx, Lx], [lower[1, -1], upper[1, -1]]])
    fault_down = _fault_segment(
        upper[1, 0], lower[1, 0], Lx, Ly, frac_x_bottom, frac_x_top
    )
    return np.column_stack(
        (lower, right_up[:, 1:], np.fliplr(upper), fault_down[:, 1:])
    )


def _classify_layer_cells(
    sd: pg.Grid,
    n_layers: int,
    Lx: float,
    Ly: float,
    frac_x_bottom: float,
    frac_x_top: float,
    fault_displacement: float,
    left_bounds: list[ArrayF],
    right_bounds: list[ArrayF],
) -> ArrayI:
    """Classify each cell into a geological layer using polygon tests."""
    cell_layer = np.full(sd.num_cells, -1, dtype=int)

    for i in range(n_layers):
        poly_left = _left_piece_polygon(
            left_bounds[i], left_bounds[i + 1], Lx, Ly, frac_x_bottom, frac_x_top
        )
        in_left = _points_in_polygon(sd, poly_left)
        mark_left = np.logical_and(in_left, cell_layer < 0)
        cell_layer[mark_left] = i

        poly_right = _right_piece_polygon(
            right_bounds[i], right_bounds[i + 1], Lx, Ly, frac_x_bottom, frac_x_top
        )
        in_right = _points_in_polygon(sd, poly_right)
        mark_right = np.logical_and(in_right, cell_layer < 0)
        cell_layer[mark_right] = i

    # Fallback only for rare boundary points.
    if np.any(cell_layer < 0):
        x_fault_at_cell = _fault_x(
            sd.cell_centers[1], Lx, Ly, frac_x_bottom, frac_x_top
        )
        cy_eff = sd.cell_centers[1].copy()
        right_block = sd.cell_centers[0] >= x_fault_at_cell
        cy_eff[right_block] -= fault_displacement * Ly
        cy_eff = np.clip(cy_eff, 0.0, Ly - 1e-10)
        fallback = (cy_eff / Ly * n_layers).astype(int)
        cell_layer[cell_layer < 0] = fallback[cell_layer < 0]

    return cell_layer


def _noisy_ellipse_polygon(
    center: tuple[float, float],
    axes: tuple[float, float],
    n_points: int,
    noise_level: float,
    rng_seed: int,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
) -> ArrayF:
    """Create a robust noisy ellipse polygon (2xN vertices).

    The perturbation is radial in angle space to preserve vertex ordering and
    reduce self-intersection risk. If bounds are provided, points are clipped
    to remain strictly inside the domain.
    """
    if n_points < 3:
        raise ValueError("n_points must be >= 3")
    if noise_level < 0:
        raise ValueError("noise_level must be >= 0")

    a, b = axes
    if a <= 0 or b <= 0:
        raise ValueError("axes values must be > 0")

    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    cx, cy = center

    rng = np.random.default_rng(rng_seed)

    # Radial noise keeps points in angular order and is more robust than
    # independent x/y jitter for polygon quality.
    radial = 1.0 + rng.normal(0.0, noise_level, size=n_points)
    radial = np.maximum(radial, 0.05)

    x = cx + a * radial * np.cos(theta)
    y = cy + b * radial * np.sin(theta)

    if x_bounds is not None:
        eps_x = 1e-8 * max(1.0, abs(x_bounds[1] - x_bounds[0]))
        x = np.clip(x, x_bounds[0] + eps_x, x_bounds[1] - eps_x)
    if y_bounds is not None:
        eps_y = 1e-8 * max(1.0, abs(y_bounds[1] - y_bounds[0]))
        y = np.clip(y, y_bounds[0] + eps_y, y_bounds[1] - eps_y)

    return np.vstack((x, y))


def _segments_intersect(a: ArrayF, b: ArrayF, c: ArrayF, d: ArrayF) -> bool:
    """Return True if closed segments ab and cd intersect (including touching)."""

    def orient(p: ArrayF, q: ArrayF, r: ArrayF) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_seg(p: ArrayF, q: ArrayF, r: ArrayF) -> bool:
        return (
            min(p[0], r[0]) - 1e-14 <= q[0] <= max(p[0], r[0]) + 1e-14
            and min(p[1], r[1]) - 1e-14 <= q[1] <= max(p[1], r[1]) + 1e-14
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if (o1 > 0 > o2 or o1 < 0 < o2) and (o3 > 0 > o4 or o3 < 0 < o4):
        return True

    if abs(o1) <= 1e-14 and on_seg(a, c, b):
        return True
    if abs(o2) <= 1e-14 and on_seg(a, d, b):
        return True
    if abs(o3) <= 1e-14 and on_seg(c, a, d):
        return True
    if abs(o4) <= 1e-14 and on_seg(c, b, d):
        return True

    return False


def _is_simple_polygon(polygon_xy: ArrayF) -> bool:
    """Check if polygon edges do not self-intersect."""
    n = polygon_xy.shape[1]
    for i in range(n):
        i2 = (i + 1) % n
        a = polygon_xy[:, i]
        b = polygon_xy[:, i2]
        for j in range(i + 1, n):
            j2 = (j + 1) % n

            # Adjacent edges share a vertex and are allowed to touch.
            if i == j or i2 == j or j2 == i:
                continue
            # First and last edges are also adjacent in closed polygon.
            if i == 0 and j2 == 0:
                continue

            c = polygon_xy[:, j]
            d = polygon_xy[:, j2]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def _polygon_constraints(polygon_xy: ArrayF) -> list[pp.LineFracture]:
    """Convert a polygon (2xN) into closed boundary line constraints."""
    constraints: list[pp.LineFracture] = []
    n_vertices = polygon_xy.shape[1]
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        edge = np.array(
            [
                [polygon_xy[0, i], polygon_xy[0, j]],
                [polygon_xy[1, i], polygon_xy[1, j]],
            ]
        )
        constraints.append(pp.LineFracture(edge))
    return constraints


def _subgrid_inside_polygon(sd: pg.Grid, polygon_xy: ArrayF) -> pg.Grid:
    """Extract the subgrid composed by cells inside polygon_xy."""
    inside = _points_in_polygon(sd, polygon_xy).astype(int)
    subgrid = pp.partition.partition_grid(sd, inside)[0][0]
    pg.convert_from_pp(subgrid)
    subgrid.compute_geometry()
    return subgrid


def _subgrid_outside_polygon(sd: pg.Grid, polygon_xy: ArrayF) -> pg.Grid:
    """Extract the subgrid composed by cells outside polygon_xy."""
    inside = _points_in_polygon(sd, polygon_xy)
    keep_cells = np.where(~inside)[0]
    subgrid, _, _ = pp.partition.extract_subgrid(sd, keep_cells)
    pg.convert_from_pp(subgrid)
    subgrid.compute_geometry()
    return subgrid


def create_layered_grid(
    mesh_size: float,
    Lx: float = 2.0,
    Ly: float = 1.0,
    n_layers: int = 4,
    frac_x_bottom: float = 0.35,
    frac_x_top: float = 0.60,
    fault_displacement: float = 0.08,
    roughness: float = 0.02,
    n_seg: int = 5,
    rng_seed: int = 42,
) -> tuple[pg.Grid, ArrayI]:
    """Create a layered/faulted domain and return layer index for each cell."""
    if mesh_size <= 0:
        raise ValueError("mesh_size must be > 0")
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx and Ly must be > 0")
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if not (0.0 <= frac_x_bottom <= 1.0 and 0.0 <= frac_x_top <= 1.0):
        raise ValueError("frac_x_bottom and frac_x_top must be in [0, 1]")
    if roughness < 0:
        raise ValueError("roughness must be >= 0")
    if n_seg < 1:
        raise ValueError("n_seg must be >= 1")

    constraints, left_bounds, right_bounds = _build_layer_constraints(
        Lx=Lx,
        Ly=Ly,
        n_layers=n_layers,
        frac_x_bottom=frac_x_bottom,
        frac_x_top=frac_x_top,
        fault_displacement=fault_displacement,
        roughness=roughness,
        n_seg=n_seg,
        rng_seed=rng_seed,
    )

    sd = _mesh_from_domain(mesh_size, Lx, Ly, constraints)
    cell_layer = _classify_layer_cells(
        sd=sd,
        n_layers=n_layers,
        Lx=Lx,
        Ly=Ly,
        frac_x_bottom=frac_x_bottom,
        frac_x_top=frac_x_top,
        fault_displacement=fault_displacement,
        left_bounds=left_bounds,
        right_bounds=right_bounds,
    )

    return sd, cell_layer


def create_layered_grid_with_hole(
    mesh_size: float,
    Lx: float = 2.0,
    Ly: float = 1.0,
    n_layers: int = 4,
    frac_x_bottom: float = 0.35,
    frac_x_top: float = 0.60,
    fault_displacement: float = 0.08,
    roughness: float = 0.02,
    n_seg: int = 5,
    rng_seed: int = 42,
    center: tuple[float, float] | None = None,
    axes: tuple[float, float] = (0.2, 0.1),
    n_hole_points: int = 20,
    hole_noise_level: float = 0.01,
    hole_seed: int = 123,
) -> tuple[pg.Grid, ArrayI]:
    """Create layered domain and remove a hole after meshing.

    Workflow:
    1. Build layer and hole boundary constraints.
    2. Mesh the full constrained domain.
    3. Remove cells whose centers lie inside the hole polygon.
    """
    if center is None:
        center = (0.5 * Lx, 0.5 * Ly)

    constraints, left_bounds, right_bounds = _build_layer_constraints(
        Lx=Lx,
        Ly=Ly,
        n_layers=n_layers,
        frac_x_bottom=frac_x_bottom,
        frac_x_top=frac_x_top,
        fault_displacement=fault_displacement,
        roughness=roughness,
        n_seg=n_seg,
        rng_seed=rng_seed,
    )

    # Generate a robust non-self-intersecting hole polygon.
    hole_poly = None
    for attempt in range(8):
        level = hole_noise_level * (0.7**attempt)
        candidate = _noisy_ellipse_polygon(
            center=center,
            axes=axes,
            n_points=n_hole_points,
            noise_level=level,
            rng_seed=hole_seed + attempt,
            x_bounds=(0.0, Lx),
            y_bounds=(0.0, Ly),
        )
        if _is_simple_polygon(candidate):
            hole_poly = candidate
            break
    if hole_poly is None:
        # Deterministic safe fallback without noise.
        hole_poly = _noisy_ellipse_polygon(
            center=center,
            axes=axes,
            n_points=n_hole_points,
            noise_level=0.0,
            rng_seed=hole_seed,
            x_bounds=(0.0, Lx),
            y_bounds=(0.0, Ly),
        )

    # Enforce the hole boundary in the mesh, then classify with point-in-polygon.
    hole_constraints = _polygon_constraints(hole_poly)
    all_constraints = [*constraints, *hole_constraints]
    sd_layered = _mesh_from_domain(mesh_size, Lx, Ly, all_constraints)

    sd_hole = _subgrid_outside_polygon(sd_layered, hole_poly)

    cell_layer_hole = _classify_layer_cells(
        sd=sd_hole,
        n_layers=n_layers,
        Lx=Lx,
        Ly=Ly,
        frac_x_bottom=frac_x_bottom,
        frac_x_top=frac_x_top,
        fault_displacement=fault_displacement,
        left_bounds=left_bounds,
        right_bounds=right_bounds,
    )

    return sd_hole, cell_layer_hole


def layers(
    mesh_size: float,
    Lx: float = 2.0,
    Ly: float = 1.0,
    n_layers: int = 4,
    frac_x_bottom: float = 0.35,
    frac_x_top: float = 0.60,
    fault_displacement: float = 0.08,
    roughness: float = 0.02,
    n_seg: int = 5,
    rng_seed: int = 42,
) -> tuple[pg.Grid, ArrayI]:
    """Backward-compatible alias for create_layered_grid."""
    return create_layered_grid(
        mesh_size=mesh_size,
        Lx=Lx,
        Ly=Ly,
        n_layers=n_layers,
        frac_x_bottom=frac_x_bottom,
        frac_x_top=frac_x_top,
        fault_displacement=fault_displacement,
        roughness=roughness,
        n_seg=n_seg,
        rng_seed=rng_seed,
    )


def create_grid(mesh_size: float) -> pg.Grid:
    """Create a 2D unit-square grid with computed geometry."""
    if mesh_size <= 0:
        raise ValueError("mesh_size must be > 0")

    sd = pg.unit_grid(2, mesh_size, as_mdg=False)
    sd.compute_geometry()
    return sd


def create_grid_with_hole(
    mesh_size: float,
    center: tuple[float, float] = (0.5, 0.5),
    axes: tuple[float, float] = (0.2, 0.1),
    n_points: int = 20,
    noise_level: float = 0.01,
    rng_seed: int = 42,
) -> pg.Grid:
    """Backward-compatible hole grid on unit square.

    Internally built from the layered+hole pipeline with one flat layer and no fault
    displacement, then returns only the grid for compatibility with existing code.
    """
    sd_hole, _ = create_layered_grid_with_hole(
        mesh_size=mesh_size,
        Lx=1.0,
        Ly=1.0,
        n_layers=1,
        frac_x_bottom=0.5,
        frac_x_top=0.5,
        fault_displacement=0.0,
        roughness=0.0,
        n_seg=2,
        rng_seed=0,
        center=center,
        axes=axes,
        n_hole_points=n_points,
        hole_noise_level=noise_level,
        hole_seed=rng_seed,
    )
    return sd_hole
