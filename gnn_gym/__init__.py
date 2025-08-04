"""
GNN gym

Exercise with examples how a graph neural network (GNN) works
"""

import argparse
import sys

from .test_cases import load_test_case, execute_test_case

def main() -> None:
    """
    Main entry point
    """

    parser = argparse.ArgumentParser(description="Run a GNN test case dynamically.")
    parser.add_argument(
        "--test_case",
        help="Name of the test case (e.g., 'small_graph', 'cora', 'case_link_pred')"
    )
    parser.add_argument("--override", nargs="*", help="Key=Value pairs to override config")
    opts = parser.parse_args()

    # load module
    try:
        module = load_test_case(case_name=opts.test_case)
    except ModuleNotFoundError as e:
        print(f"{e}")
        sys.exit(1)

    # parse overrides
    config_override = {}
    if opts.override:
        for item in opts.override:
            if "=" not in item:
                print(f"⚠️ Invalid override format: {item}")
                continue
            key, value = item.split("=", 1)
            # try to cast to int/float, fallback to string
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass
            config_override[key] = value

    execute_test_case(module, config_override)
