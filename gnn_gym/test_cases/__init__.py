
import importlib
from types import ModuleType

# loader
def load_test_case(case_name: str) -> ModuleType:
    """Dynamically load a test case module by name.
    """
    module_names = [
        f".test_cases.{case_name}",
        f".test_cases.case_{case_name}"
    ]
    for mod_name in module_names:
        try:
            return importlib.import_module(mod_name, package="gnn_gym")
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(f"Test case module '{case_name}' not found.")

# runner
def execute_test_case(module, config_override=None):
    """
    Generic function to run any test case module that follows the interface.
    """
    header = f"\n- Running pipeline for test case: {module.__name__}"
    print(header)
    print("=" * len(header))

    # run test case
    try:
        results = module.run(config_override=config_override)
        print(f"\n- Test completed. Results: {results}")
    except Exception as e:
        print(f"\n- Error during GNN execution: {e}")
