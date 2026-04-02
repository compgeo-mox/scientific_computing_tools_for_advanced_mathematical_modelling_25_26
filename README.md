# Scientific Computing Tools for Advanced Mathematical Modelling (2025-26)

This repository contains small teaching examples for:

1. 2-D grid generation with PorePy/PyGeoN
2. Layered and faulted geological domains
3. Hole carving through polygon-based subgrid extraction
4. Linear elasticity assembly, solve, and export

## Project Structure

- [src/create_grid.py](src/create_grid.py): grid generators (plain, layered, hole, layered+hole)
- [src/elastic_pb.py](src/elastic_pb.py): elasticity assembly/solve/export helper class
- [src/common_functions.py](src/common_functions.py): shared solve+export workflow
- [src/main.py](src/main.py): CLI entry point for selecting grid case
- [src/main_unit_grid.py](src/main_unit_grid.py): unit square case
- [src/main_unit_grid_hole.py](src/main_unit_grid_hole.py): unit square with hole case
- [src/main_layered_grid.py](src/main_layered_grid.py): layered/faulted case
- [src/main_layered_grid_hole.py](src/main_layered_grid_hole.py): layered/faulted with hole case
- [src/results](src/results): VTU/PVD outputs for visualization

## Run

From repository root:

```bash
python src/main.py
```

By default this runs all cases.

You can select one case with:

```bash
python src/main.py --case unit
python src/main.py --case unit_hole
python src/main.py --case layered
python src/main.py --case layered_hole
python src/main.py --case all
```

You can also run dedicated entry points directly:

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

- The scripts assume required Python packages are available in the interpreter used to run them.
- If one interpreter fails with missing dependencies, run with an environment where `numpy`, `porepy`, and `pygeon` are installed.