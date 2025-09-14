import json
import os

# Path to the JSON file containing model keys
KEY_FILE_PATH = os.path.join(os.path.dirname(__file__), "keys.json")


def get_model_keys(model_name: str) -> tuple[str]:
    """
    Retrieve the configuration keys for the given model name from keys.json.

    :param model_name: Name of the model
    :return: Tuple of expected configuration keys
    :raises FileNotFoundError: If the keys.json file does not exist
    :raises ValueError: If the model name is not found in keys.json
    """
    if not os.path.exists(KEY_FILE_PATH):
        raise FileNotFoundError(f"keys.json not found at {KEY_FILE_PATH}")

    with open(KEY_FILE_PATH, "r") as f:
        keys_data = json.load(f)

    if model_name not in keys_data:
        raise ValueError(f"Model name '{model_name}' not found in keys.json")

    return keys_data[model_name]
