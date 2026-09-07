import argparse
import json


def json_loads_dict(s: str) -> dict:
    """Parses a JSON string into a dictionary.

    Args:
        s (str): The JSON string to parse.

    Returns:
        dict: The parsed dictionary.

    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid JSON dictionary.
    """
    d = json.loads(s)
    if isinstance(d, dict):
        return d
    raise argparse.ArgumentTypeError(f"Expected a JSON dictionary, got {type(d)}")
