import argparse
from pathlib import Path
import shutil
import json

TEMPLATE_DIR = Path(__file__).parent
SRC_TEMPLATE = """from template_nn.networks.base_nn import BaseNetwork
from ..nn_keys import MODEL_KEYS


class {name}(BaseNetwork):
    def __init__(self, tabular: dict, visualise: bool = False) -> None:
        super().__init__(tabular, model_keys=MODEL_KEYS["{name}"], visualise=visualise)

    def _create_layers(self, *args, **kwargs):
        raise NotImplementedError("Define layer structure here")

    def _build_model(self, *args, **kwargs):
        raise NotImplementedError("Define how model is built here")
"""

TEST_BUILD_TEMPLATE = """from template_nn import {name}


def test_build_model():
    model = {name}({{}}, visualise=True)
    assert model is not None
"""

TEST_CREATE_TEMPLATE = """from template_nn import {name}


def test_create_layers():
    model = {name}({{}}, visualise=True)
    assert hasattr(model, "_create_layers")
"""


def snake_case(name):
    return name.lower()


def create_network_files(name):
    snake = snake_case(name)
    test_dir = TEMPLATE_DIR / "tests" / "networks" / f"test_{snake}"
    src_file = TEMPLATE_DIR / "src" / "template_nn" / "networks" / f"{snake}.py"

    test_dir.mkdir(parents=True, exist_ok=True)
    src_file.parent.mkdir(parents=True, exist_ok=True)

    with open(src_file, "w") as f:
        f.write(SRC_TEMPLATE.format(name=name))

    with open(test_dir / f"test_{snake}_build_model.py", "w") as f:
        f.write(TEST_BUILD_TEMPLATE.format(name=name))

    with open(test_dir / f"test_{snake}_create_layers.py", "w") as f:
        f.write(TEST_CREATE_TEMPLATE.format(name=name))

    nn_keys_path = TEMPLATE_DIR / "src" / "template_nn" / "keys.json"
    if nn_keys_path.exists():
        with open(nn_keys_path, "r") as f:
            keys_data = json.load(f)
    else:
        keys_data = {}

    keys_data[name] = []

    with open(nn_keys_path, "w") as f:
        json.dump(keys_data, f, indent=4)

    print(f"Created network stub and tests for: {name}")


def remove_network_files(name):
    snake = snake_case(name)
    test_dir = TEMPLATE_DIR / "tests" / "networks" / f"test_{snake}"
    src_file = TEMPLATE_DIR / "src" / "template_nn" / "networks" / f"{snake}.py"
    nn_keys_path = TEMPLATE_DIR / "src" / "template_nn" / "keys.json"

    if src_file.exists():
        src_file.unlink()
        print(f"Removed source file: {src_file}")
    else:
        print(f"Source file not found: {src_file}")

    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir)
        print(f"Removed test directory: {test_dir}")
    else:
        print(f"Test directory not found: {test_dir}")

    if nn_keys_path.exists():
        with open(nn_keys_path, "r") as f:
            keys_data = json.load(f)
        if name in keys_data:
            del keys_data[name]
            with open(nn_keys_path, "w") as f:
                json.dump(keys_data, f, indent=4)
            print(f"Removed {name} from keys.json")
        else:
            print(f"{name} not found in keys.json")
    else:
        print("keys.json not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create or remove boilerplate for a network."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--install", help="Name of the network to install (e.g. RNN)")
    group.add_argument("--remove", help="Name of the network to remove (e.g. RNN)")

    args = parser.parse_args()

    if args.install:
        create_network_files(args.install)
    elif args.remove:
        remove_network_files(args.remove)
