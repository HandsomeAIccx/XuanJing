import numpy as np
from typing import List, Tuple, Dict


def stack_dict(data: List[Dict], keys: List[str]) -> Dict:
    """
    Stack the given data for each key into a dictionary.

    Args:
        data: A list of dictionaries to stack
        keys: A list of keys to stack the data for

    Returns:
        A dictionary with the stacked data for each key
    """
    # Check if the input is a list or tuple
    assert isinstance(data, (list, tuple)), f"Given Data Type {type(data)} is not support now!"
    # Create a new dictionary with the stacked values
    res = {key: np.stack([item[key] for item in data]) for key in keys}
    # Return the dictionary
    return res
