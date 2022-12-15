import os
import json
import torch
import numpy as np


def tensorify(data):
    if isinstance(data, dict):
        return {k: tensorify(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        if data.dtype in [np.int64, int]:
            return torch.tensor(data, dtype=torch.int64)
        else:
            return torch.tensor(data, dtype=torch.float)
    elif isinstance(data, list):
        return [torch.tensor(i) for i in data]
    else:
        return ValueError(f"Not Support type {type(data)}")


def numpyify(data):
    if isinstance(data, dict):
        return {k: numpyify(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [numpyify(i) for i in data]
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return np.asarray(data)


def dump_json_data(data, file_name):
    assert type(file_name) == str, "file_path must be a string."
    assert file_name.endswith('.json'), "file_path must endswith '.json'."
    current_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(current_path, file_name)
    if not os.path.exists(file_path):
        os.system(f"touch {file_path}")
    with open(file_path, 'a') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=1)


def deep_list(data, dtype=None):
    if isinstance(data, dict):
        return {k: deep_list(v, dtype) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().tolist()
    else:
        raise ValueError(f"Not Support type {type(data)}.")