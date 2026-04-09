import numpy as np

import pygeon as pg
import porepy as pp

"""Grid generation utilities for layered and holed 2-D domains.

Public functions are grouped at the top of the file.
Private helpers used by the workflow are defined below.
"""


def create_layered_grid(
    mesh_size,
    Lx=2.0,
    Ly=1.0,
    n_layers=4,
    frac_x_bottom=0.35,
    frac_x_top=0.60,
    fault_displacement=0.08,
    roughness=0.02,
    n_seg=5,
    rng_seed=42,
):
    """Create a layered/faulted 2-D domain and classify cells by layer.

    Parameters
    ----------
    mesh_size : float
        Target mesh size used by the grid generator.
    Lx : float, optional
        Domain length along x.
    Ly : float, optional
        Domain length along y.
    n_layers : int, optional
        Number of geological layers.
    frac_x_bottom : float, optional
        Fault x-position fraction at ``y = 0``.
    frac_x_top : float, optional
        Fault x-position fraction at ``y = Ly``.
    fault_displacement : float, optional
        Vertical offset applied to right-block interfaces, as a fraction of
        ``Ly``.
    roughness : float, optional
        Standard deviation for random perturbations of interface points.
    n_seg : int, optional
        Number of polyline segments used per interface side.
    rng_seed : int, optional
        Seed controlling random interface perturbations.

    Returns
    -------
    tuple[pg.Grid, np.ndarray]
        ``(sd, cell_layer)`` where ``sd`` is the generated grid and
        ``cell_layer`` stores one integer layer id per cell.
    """
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

    # Step 1: build the fault line plus all rough layer interfaces.
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

    # Step 2: generate the constrained mesh in the physical domain.
    sd = _mesh_from_domain(mesh_size, Lx, Ly, constraints)

    # Step 3: classify each mesh cell by point-in-polygon layer tests.
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
    mesh_size,
    Lx=2.0,
    Ly=1.0,
    n_layers=4,
    frac_x_bottom=0.35,
    frac_x_top=0.60,
    fault_displacement=0.08,
    roughness=0.02,
    n_seg=5,
    rng_seed=42,
    center=None,
    axes=(0.2, 0.1),
    n_hole_points=20,
    hole_noise_level=0.01,
    hole_seed=123,
):
    """Create a layered/faulted grid and carve a noisy elliptical hole.

    Parameters
    ----------
    mesh_size : float
        Target mesh size used by the grid generator.
    Lx : float, optional
        Domain length along x.
    Ly : float, optional
        Domain length along y.
    n_layers : int, optional
        Number of geological layers.
    frac_x_bottom : float, optional
        Fault x-position fraction at ``y = 0``.
    frac_x_top : float, optional
        Fault x-position fraction at ``y = Ly``.
    fault_displacement : float, optional
        Vertical offset applied to right-block interfaces, as a fraction of
        ``Ly``.
    roughness : float, optional
        Standard deviation for random perturbations of interface points.
    n_seg : int, optional
        Number of polyline segments used per interface side.
    rng_seed : int, optional
        Seed controlling random interface perturbations.
    center : tuple[float, float] or None, optional
        Hole center. If ``None``, the domain center is used.
    axes : tuple[float, float], optional
        Hole semi-axes ``(a, b)``.
    n_hole_points : int, optional
        Number of polygon vertices used to represent the hole.
    hole_noise_level : float, optional
        Relative radial noise amplitude for hole-shape perturbation.
    hole_seed : int, optional
        Seed controlling hole-shape perturbations.

    Returns
    -------
    tuple[pg.Grid, np.ndarray]
        ``(sd_hole, cell_layer_hole)`` where ``sd_hole`` excludes cells inside
        the hole and ``cell_layer_hole`` contains layer ids for remaining cells.
    """
    if center is None:
        center = (0.5 * Lx, 0.5 * Ly)

    # Step 1: build layered constraints (fault + rough interfaces).
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

    # Step 2: build a robust hole polygon; if needed, reduce noise gradually.
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

    # Safety fallback: use a deterministic ellipse with no noise.
    if hole_poly is None:
        hole_poly = _noisy_ellipse_polygon(
            center=center,
            axes=axes,
            n_points=n_hole_points,
            noise_level=0.0,
            rng_seed=hole_seed,
            x_bounds=(0.0, Lx),
            y_bounds=(0.0, Ly),
        )

    # Step 3: mesh with hole boundary constraints and drop inside-hole cells.
    hole_constraints = _polygon_constraints(hole_poly)
    all_constraints = [*constraints, *hole_constraints]
    sd_layered = _mesh_from_domain(mesh_size, Lx, Ly, all_constraints)
    sd_hole = _subgrid_outside_polygon(sd_layered, hole_poly)

    # Step 4: classify only the cells that remain after hole extraction.
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
    mesh_size,
    Lx=2.0,
    Ly=1.0,
    n_layers=4,
    frac_x_bottom=0.35,
    frac_x_top=0.60,
    fault_displacement=0.08,
    roughness=0.02,
    n_seg=5,
    rng_seed=42,
):
    """Backward-compatible alias of :func:`create_layered_grid`.

    Parameters
    ----------
    mesh_size : float
        Target mesh size used by the grid generator.
    Lx : float, optional
        Domain length along x.
    Ly : float, optional
        Domain length along y.
    n_layers : int, optional
        Number of geological layers.
    frac_x_bottom : float, optional
        Fault x-position fraction at ``y = 0``.
    frac_x_top : float, optional
        Fault x-position fraction at ``y = Ly``.
    fault_displacement : float, optional
        Vertical offset applied to right-block interfaces, as a fraction of
        ``Ly``.
    roughness : float, optional
        Standard deviation for random perturbations of interface points.
    n_seg : int, optional
        Number of polyline segments used per interface side.
    rng_seed : int, optional
        Seed controlling random interface perturbations.

    Returns
    -------
    tuple[pg.Grid, np.ndarray]
        Same return tuple as :func:`create_layered_grid`, ``(sd, cell_layer)``.
    """
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


def create_grid(mesh_size):
    """Create a simple unit-square grid.

    Parameters
    ----------
    mesh_size : float
        Target mesh size controlling grid resolution.

    Returns
    -------
    pg.Grid
        Two-dimensional unit-square grid with geometry precomputed.
    """
    if mesh_size <= 0:
        raise ValueError("mesh_size must be > 0")

    # Build unit grid and precompute geometry for immediate use.
    sd = pg.unit_grid(2, mesh_size, as_mdg=False)
    sd.compute_geometry()
    return sd


def create_grid_with_hole(
    mesh_size,
    center=(0.5, 0.5),
    axes=(0.2, 0.1),
    n_points=20,
    noise_level=0.01,
    rng_seed=42,
):
    """Create a unit-square grid with a noisy elliptical hole.

    Parameters
    ----------
    mesh_size : float
        Target mesh size used by the grid generator.
    center : tuple[float, float], optional
        Hole center in unit-square coordinates.
    axes : tuple[float, float], optional
        Hole semi-axes ``(a, b)``.
    n_points : int, optional
        Number of polygon vertices used to represent the hole.
    noise_level : float, optional
        Relative radial noise amplitude for hole-shape perturbation.
    rng_seed : int, optional
        Seed controlling hole-shape perturbations.

    Returns
    -------
    pg.Grid
        Unit-square grid with cells inside the hole removed.
    """
    # Reuse the layered pipeline with a single flat layer and no fault throw.
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


def _polygon_to_3d(vertices_xy):
    """Convert a 2xN polygon to PorePy's 3xN format.

    Parameters
    ----------
    vertices_xy : np.ndarray
        Polygon vertices in 2-D, shape (2, N).

    Returns
    -------
    poly_3d : np.ndarray
        Same polygon in 3-D format (z=0), shape (3, N).
    """
    # PorePy polygon utilities expect 3 coordinates per vertex.
    return np.vstack((vertices_xy, np.zeros((1, vertices_xy.shape[1]))))


def _remove_duplicate_vertices(vertices_xy, tol=1e-12):
    """Remove consecutive duplicate polygon vertices.

    Parameters
    ----------
    vertices_xy : np.ndarray
        Polygon vertices in 2-D, shape (2, N).
    tol : float
        Distance tolerance to detect duplicates.

    Returns
    -------
    cleaned : np.ndarray
        Polygon vertices without repeated consecutive points.
    """
    # Keep the first vertex and filter out near-identical consecutive points.
    keep = np.ones(vertices_xy.shape[1], dtype=bool)
    diffs = np.linalg.norm(np.diff(vertices_xy, axis=1), axis=0)
    keep[1:] = diffs > tol
    cleaned = vertices_xy[:, keep]

    # If first and last are effectively equal, drop the last one.
    if cleaned.shape[1] >= 2 and np.linalg.norm(cleaned[:, 0] - cleaned[:, -1]) <= tol:
        cleaned = cleaned[:, :-1]

    return cleaned


def _points_in_polygon(sd, polygon_xy):
    """Check which cell centers are inside a polygon.

    Parameters
    ----------
    sd : pg.Grid
        Grid whose cell centers are tested.
    polygon_xy : np.ndarray
        Polygon vertices in 2-D, shape (2, N).

    Returns
    -------
    in_poly : np.ndarray
        Boolean mask of length sd.num_cells. True means inside polygon.
    """
    # Pre-clean and convert polygon so the geometry predicate is robust.
    poly = _polygon_to_3d(_remove_duplicate_vertices(polygon_xy))
    _, _, in_poly = pp.distances.points_polygon(sd.cell_centers, poly)
    return in_poly


def _mesh_from_domain(mesh_size, Lx, Ly, constraints):
    """Build a constrained 2-D mesh in [0, Lx] x [0, Ly].

    Parameters
    ----------
    mesh_size : float
        Target mesh size used by the grid generator.
    Lx : float
        Domain length in x direction.
    Ly : float
        Domain length in y direction.
    constraints : list
        List of pp.LineFracture constraints to enforce during meshing.

    Returns
    -------
    sd : pg.Grid
        Constrained grid with computed geometry.
    """
    # Define physical box and mesh while enforcing all line constraints.
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


def _fault_x(y, Lx, Ly, xb, xt):
    """Compute fault x-position at height y.

    Parameters
    ----------
    y : float or np.ndarray
        Vertical coordinate(s) where fault position is evaluated.
    Lx : float
        Domain length in x direction.
    Ly : float
        Domain length in y direction.
    xb : float
        Fault x-position fraction at y=0.
    xt : float
        Fault x-position fraction at y=Ly.

    Returns
    -------
    x_fault : float or np.ndarray
        Fault x-position(s) in physical coordinates.
    """
    # Linear interpolation of fault location from bottom to top.
    return (xb + (xt - xb) * (np.asarray(y) / Ly)) * Lx


def _fault_segment(y0, y1, Lx, Ly, frac_x_bottom, frac_x_top, n=5):
    """Build a polyline along the fault between two y-levels.

    Parameters
    ----------
    y0 : float
        Start y-coordinate.
    y1 : float
        End y-coordinate.
    Lx : float
        Domain length in x direction.
    Ly : float
        Domain length in y direction.
    frac_x_bottom : float
        Fault x-position fraction at y=0.
    frac_x_top : float
        Fault x-position fraction at y=Ly.
    n : int
        Number of interpolation points on the segment.

    Returns
    -------
    seg : np.ndarray
        Fault polyline points, shape (2, n).
    """
    # Sample y values and project them onto the slanted fault line.
    y_vals = np.linspace(y0, y1, n)
    x_vals = _fault_x(y_vals, Lx, Ly, frac_x_bottom, frac_x_top)
    return np.vstack((x_vals, y_vals))


def _noisy_polyline(x0, x1, y_mean, n_seg, roughness, rng):
    """Create a noisy horizontal polyline and convert it to line constraints.

    Parameters
    ----------
    x0 : float
        Left endpoint x-coordinate.
    x1 : float
        Right endpoint x-coordinate.
    y_mean : float
        Mean y-level for the interface.
    n_seg : int
        Number of segments in the polyline.
    roughness : float
        Standard deviation for y perturbations of interior points.
    rng : np.random.Generator
        Random generator used for reproducible noise.

    Returns
    -------
    segments : list
        List of pp.LineFracture segments.
    polyline : np.ndarray
        Polyline vertices, shape (2, n_seg + 1).
    """
    # Start from an equally spaced horizontal polyline.
    x_pts = np.linspace(x0, x1, n_seg + 1)
    y_pts = np.full(n_seg + 1, y_mean)

    # Perturb only interior points to keep endpoints anchored.
    if n_seg > 1:
        y_pts[1:-1] += rng.normal(0.0, roughness, size=n_seg - 1)

    # Convert each polyline edge into a meshing constraint.
    segments = []
    for k in range(n_seg):
        seg = np.array([[x_pts[k], x_pts[k + 1]], [y_pts[k], y_pts[k + 1]]])
        segments.append(pp.LineFracture(seg))

    return segments, np.vstack((x_pts, y_pts))


def _build_layer_constraints(
    Lx,
    Ly,
    n_layers,
    frac_x_bottom,
    frac_x_top,
    fault_displacement,
    roughness,
    n_seg,
    rng_seed,
):
    """Build fault and rough layer-interface constraints.

    Parameters
    ----------
    Lx : float
        Domain length in x direction.
    Ly : float
        Domain length in y direction.
    n_layers : int
        Number of geological layers.
    frac_x_bottom : float
        Fault x-position fraction at y=0.
    frac_x_top : float
        Fault x-position fraction at y=Ly.
    fault_displacement : float
        Vertical displacement on the right block, fraction of Ly.
    roughness : float
        Standard deviation used to perturb layer interfaces.
    n_seg : int
        Number of line segments per interface side.
    rng_seed : int
        Seed for reproducible random perturbations.

    Returns
    -------
    constraints : list
        List of pp.LineFracture constraints (fault + all rough interfaces).
    left_bounds : list
        Boundary polylines for left block layer interfaces.
    right_bounds : list
        Boundary polylines for right block layer interfaces.
    """
    rng = np.random.default_rng(rng_seed)

    constraints = []
    left_bounds = [np.empty((2, 0)) for _ in range(n_layers + 1)]
    right_bounds = [np.empty((2, 0)) for _ in range(n_layers + 1)]

    x_fault_bottom = frac_x_bottom * Lx
    x_fault_top = frac_x_top * Lx

    # Bottom/top boundaries are straight and known exactly.
    left_bounds[0] = np.array([[0.0, x_fault_bottom], [0.0, 0.0]])
    left_bounds[n_layers] = np.array([[0.0, x_fault_top], [Ly, Ly]])
    right_bounds[0] = np.array([[x_fault_bottom, Lx], [0.0, 0.0]])
    right_bounds[n_layers] = np.array([[x_fault_top, Lx], [Ly, Ly]])

    # Main fault line used as geometric discontinuity.
    constraints.append(
        pp.LineFracture(np.array([[x_fault_bottom, x_fault_top], [0.0, Ly]]))
    )

    # Internal interfaces are rough and shifted on the right block.
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


def _left_piece_polygon(lower, upper, Lx, Ly, frac_x_bottom, frac_x_top):
    """Build closed polygon for one left-block layer slice.

    Parameters
    ----------
    lower : np.ndarray
        Lower interface polyline, shape (2, N).
    upper : np.ndarray
        Upper interface polyline, shape (2, M).
    Lx : float
        Domain length in x direction.
    Ly : float
        Domain length in y direction.
    frac_x_bottom : float
        Fault x-position fraction at y=0.
    frac_x_top : float
        Fault x-position fraction at y=Ly.

    Returns
    -------
    polygon_xy : np.ndarray
        Closed polygon vertices describing one left block piece.
    """
    # Connect lower and upper boundaries with fault side and left wall side.
    fault_up = _fault_segment(
        lower[1, -1], upper[1, -1], Lx, Ly, frac_x_bottom, frac_x_top
    )
    left_down = np.array([[0.0, 0.0], [upper[1, 0], lower[1, 0]]])
    return np.column_stack((lower, fault_up[:, 1:], np.fliplr(upper), left_down[:, 1:]))


def _right_piece_polygon(lower, upper, Lx, Ly, frac_x_bottom, frac_x_top):
    """Build closed polygon for one right-block layer slice.

    Parameters
    ----------
    lower : np.ndarray
        Lower interface polyline, shape (2, N).
    upper : np.ndarray
        Upper interface polyline, shape (2, M).
    Lx : float
        Domain length in x direction.
    Ly : float
        Domain length in y direction.
    frac_x_bottom : float
        Fault x-position fraction at y=0.
    frac_x_top : float
        Fault x-position fraction at y=Ly.

    Returns
    -------
    polygon_xy : np.ndarray
        Closed polygon vertices describing one right block piece.
    """
    # Connect lower and upper boundaries with right wall side and fault side.
    right_up = np.array([[Lx, Lx], [lower[1, -1], upper[1, -1]]])
    fault_down = _fault_segment(
        upper[1, 0], lower[1, 0], Lx, Ly, frac_x_bottom, frac_x_top
    )
    return np.column_stack(
        (lower, right_up[:, 1:], np.fliplr(upper), fault_down[:, 1:])
    )


def _classify_layer_cells(
    sd,
    n_layers,
    Lx,
    Ly,
    frac_x_bottom,
    frac_x_top,
    fault_displacement,
    left_bounds,
    right_bounds,
):
    """Assign a geological layer id to each grid cell.

    Parameters
    ----------
    sd : pg.Grid
        Grid to classify.
    n_layers : int
        Number of geological layers.
    Lx : float
        Domain length in x direction.
    Ly : float
        Domain length in y direction.
    frac_x_bottom : float
        Fault x-position fraction at y=0.
    frac_x_top : float
        Fault x-position fraction at y=Ly.
    fault_displacement : float
        Vertical displacement on right block, fraction of Ly.
    left_bounds : list
        Left block interface polylines.
    right_bounds : list
        Right block interface polylines.

    Returns
    -------
    cell_layer : np.ndarray
        Integer layer id for each cell.
    """
    cell_layer = np.full(sd.num_cells, -1, dtype=int)

    # Mark cells layer by layer using polygon containment tests.
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

    # Rare unassigned boundary points are handled with a geometric fallback.
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
    center,
    axes,
    n_points,
    noise_level,
    rng_seed,
    x_bounds=None,
    y_bounds=None,
):
    """Create a noisy ellipse polygon with robust vertex ordering.

    Parameters
    ----------
    center : tuple[float, float]
        Ellipse center.
    axes : tuple[float, float]
        Ellipse semi-axes (a, b).
    n_points : int
        Number of polygon vertices.
    noise_level : float
        Relative standard deviation of radial noise.
    rng_seed : int
        Seed for reproducible random perturbations.
    x_bounds : tuple[float, float] | None
        Optional min/max bounds for x clipping.
    y_bounds : tuple[float, float] | None
        Optional min/max bounds for y clipping.

    Returns
    -------
    polygon_xy : np.ndarray
        Noisy ellipse vertices, shape (2, n_points).
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

    # Radial noise preserves angular ordering better than independent x/y jitter.
    radial = 1.0 + rng.normal(0.0, noise_level, size=n_points)
    radial = np.maximum(radial, 0.05)

    x = cx + a * radial * np.cos(theta)
    y = cy + b * radial * np.sin(theta)

    # Optional clipping keeps vertices safely inside target bounds.
    if x_bounds is not None:
        eps_x = 1e-8 * max(1.0, abs(x_bounds[1] - x_bounds[0]))
        x = np.clip(x, x_bounds[0] + eps_x, x_bounds[1] - eps_x)
    if y_bounds is not None:
        eps_y = 1e-8 * max(1.0, abs(y_bounds[1] - y_bounds[0]))
        y = np.clip(y, y_bounds[0] + eps_y, y_bounds[1] - eps_y)

    return np.vstack((x, y))


def _segments_intersect(a, b, c, d):
    """Check if closed segments ab and cd intersect (touching included).

    Parameters
    ----------
    a, b, c, d : np.ndarray
        Segment endpoints, each shape (2,).

    Returns
    -------
    intersects : bool
        True if the segments intersect or touch.
    """

    def orient(p, q, r):
        # Signed area test: sign gives orientation of the triplet.
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_seg(p, q, r):
        # Bounding-box test for collinear points.
        return (
            min(p[0], r[0]) - 1e-14 <= q[0] <= max(p[0], r[0]) + 1e-14
            and min(p[1], r[1]) - 1e-14 <= q[1] <= max(p[1], r[1]) + 1e-14
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    # Proper crossing case.
    if (o1 > 0 > o2 or o1 < 0 < o2) and (o3 > 0 > o4 or o3 < 0 < o4):
        return True

    # Collinear touching cases.
    if abs(o1) <= 1e-14 and on_seg(a, c, b):
        return True
    if abs(o2) <= 1e-14 and on_seg(a, d, b):
        return True
    if abs(o3) <= 1e-14 and on_seg(c, a, d):
        return True
    if abs(o4) <= 1e-14 and on_seg(c, b, d):
        return True

    return False


def _is_simple_polygon(polygon_xy):
    """Check whether polygon edges are free of self-intersections.

    Parameters
    ----------
    polygon_xy : np.ndarray
        Polygon vertices in 2-D, shape (2, N).

    Returns
    -------
    is_simple : bool
        True if no non-adjacent edges intersect.
    """
    n = polygon_xy.shape[1]

    # Compare every edge pair except adjacent pairs that share a vertex.
    for i in range(n):
        i2 = (i + 1) % n
        a = polygon_xy[:, i]
        b = polygon_xy[:, i2]
        for j in range(i + 1, n):
            j2 = (j + 1) % n

            if i == j or i2 == j or j2 == i:
                continue
            if i == 0 and j2 == 0:
                continue

            c = polygon_xy[:, j]
            d = polygon_xy[:, j2]
            if _segments_intersect(a, b, c, d):
                return False

    return True


def _polygon_constraints(polygon_xy):
    """Convert polygon edges to a list of line constraints.

    Parameters
    ----------
    polygon_xy : np.ndarray
        Polygon vertices in 2-D, shape (2, N).

    Returns
    -------
    constraints : list
        List of pp.LineFracture edges forming a closed polygon.
    """
    constraints = []
    n_vertices = polygon_xy.shape[1]

    # Create one line constraint per edge, wrapping at the end.
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


def _subgrid_inside_polygon(sd, polygon_xy):
    """Extract a subgrid made of cells whose centers are inside a polygon.

    Parameters
    ----------
    sd : pg.Grid
        Input grid.
    polygon_xy : np.ndarray
        Polygon vertices in 2-D, shape (2, N).

    Returns
    -------
    subgrid : pg.Grid
        Subgrid containing inside cells.
    """
    # Partition helper expects an integer mask.
    inside = _points_in_polygon(sd, polygon_xy).astype(int)
    subgrid = pp.partition.partition_grid(sd, inside)[0][0]
    pg.convert_from_pp(subgrid)
    subgrid.compute_geometry()
    return subgrid


def _subgrid_outside_polygon(sd, polygon_xy):
    """Extract a subgrid made of cells whose centers are outside a polygon.

    Parameters
    ----------
    sd : pg.Grid
        Input grid.
    polygon_xy : np.ndarray
        Polygon vertices in 2-D, shape (2, N).

    Returns
    -------
    subgrid : pg.Grid
        Subgrid containing outside cells.
    """
    inside = _points_in_polygon(sd, polygon_xy)

    # Keep only cells not inside the polygon, then extract a new grid.
    keep_cells = np.where(~inside)[0]
    subgrid, _, _ = pp.partition.extract_subgrid(sd, keep_cells)
    pg.convert_from_pp(subgrid)
    subgrid.compute_geometry()
    return subgrid
