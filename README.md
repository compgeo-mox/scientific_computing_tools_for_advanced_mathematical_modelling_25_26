# Scientific Computing Tools for Advanced Mathematical Modelling (2025-26)

This repository contains small teaching examples for:

1. 2-D grid generation with PorePy/PyGeoN
2. Layered and faulted geological domains
3. Hole carving through polygon-based subgrid extraction
4. Linear elasticity assembly, solve, and export

## Installation

From repository root:

```bash
pip install .
```

For development/editable install:

```bash
pip install -e .
```

Current package setting in `pyproject.toml`: `pygeon` is installed from GitHub `main`
(`pygeon[development,testing] @ git+https://github.com/compgeo-mox/pygeon.git@main`).

## Project Structure

- [src/create_grid.py](src/create_grid.py): grid generators (plain, layered, hole, layered+hole)
- [src/elastic_pb.py](src/elastic_pb.py): elasticity assembly/solve/export helper class
- [src/common_functions.py](src/common_functions.py): shared solve+export workflow
- [src/main_unit_grid.py](src/main_unit_grid.py): unit square case
- [src/main_unit_grid_hole.py](src/main_unit_grid_hole.py): unit square with hole case
- [src/main_layered_grid.py](src/main_layered_grid.py): layered/faulted case
- [src/main_layered_grid_hole.py](src/main_layered_grid_hole.py): layered/faulted with hole case
- [src/results](src/results): VTU/PVD outputs for visualization

## Run

From repository root:

```bash
python src/main_unit_grid.py
python src/main_unit_grid_hole.py
python src/main_layered_grid.py
python src/main_layered_grid_hole.py
```

## Outputs

Each case writes displacement and optional extra fields to [src/results](src/results).

- Displacement is exported as point field `u`.
- Layered cases also export cell field `layer`.

Open the generated `.pvd` file in ParaView and color by `u` or `layer`.

## Notes

- Use the same Python environment for installation and execution.
- If dependency resolution fails, make sure `git` is available (needed for the `pygeon` GitHub dependency).