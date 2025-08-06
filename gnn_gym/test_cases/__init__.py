
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
    header = f"\n- Running test case: {module.__name__}"
    print(header)
    print("=" * len(header))

    # Step 1: Get configuration
    config = module.get_config()
    if config_override:
        config.update(config_override)

    print("- Test Case Info:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Step 2: Run the test case (GNN creation/training in future)
    print("\n- Running GNN pipeline...")
    try:
        results = module.run(config=config)
        print(f"\n- Test completed. Results: {results}")
    except Exception as e:
        print(f"\n- Error during GNN execution: {e}")
