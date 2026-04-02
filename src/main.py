"""Single command-line entry point for all elasticity grid cases.

Usage examples:
    python src/main.py
    python src/main.py --case unit
    python src/main.py --case layered_hole
"""

import argparse

from main_layered_grid import main as main_layered
from main_layered_grid_hole import main as main_layered_hole
from main_unit_grid import main as main_unit
from main_unit_grid_hole import main as main_unit_hole


def main():
    """Dispatch to a case-specific elasticity entry point.

    Input:
    Optional --case argument selecting one grid configuration.

    Output:
    Runs the selected solver workflow and writes result files.
    """
    parser = argparse.ArgumentParser(
        description="Run elasticity on a selected grid configuration."
    )
    parser.add_argument(
        "--case",
        choices=["unit", "unit_hole", "layered", "layered_hole", "all"],
        default="all",
        help="Grid configuration to run (default: all).",
    )
    args = parser.parse_args()

    # Simple case dispatcher to keep each workflow in its own module.
    if args.case == "unit":
        main_unit()
    elif args.case == "unit_hole":
        main_unit_hole()
    elif args.case == "layered":
        main_layered()
    elif args.case == "layered_hole":
        main_layered_hole()
    else:
        # Run every configuration so elasticity is solved on each grid type.
        print("Running case: unit")
        main_unit()

        print("Running case: unit_hole")
        main_unit_hole()

        print("Running case: layered")
        main_layered()

        print("Running case: layered_hole")
        main_layered_hole()


if __name__ == "__main__":
    main()
