from pathlib import Path
from typing import Dict, Tuple, Set

import numpy as np

from app.special_values import UNKNOWN_KEY, MAX_DIST, LAST_NET


def location_to_str(location):
    if location != UNKNOWN_KEY:
        return '_'.join(list(location))
    else:
        return location


def str_to_location(location):
    if location != UNKNOWN_KEY:
        return tuple(map(int, location.split('_')))
    else:
        return location


def are_points_close(letter_location, location, dist=MAX_DIST):
    return np.linalg.norm(np.array(location) - np.array(letter_location)) < dist


def is_different_values_preset(old: Dict, new: Dict) -> bool:
    return len(set(old.keys()).symmetric_difference(set(new.keys()))) > 0


def get_values_to_add_and_remove(old: Dict, new: Dict) -> Tuple[Set, Set]:
    old_keys = set(old.keys())
    new_keys = set(new.keys())
    return old_keys-new_keys, new_keys-old_keys


def get_device():
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def file_name_to_location(file_name):
    return str_to_location(file_name.name.split('.')[0])


def is_non_last_net(path: Path):
    return path.name.endswith('.pth') and path.name != LAST_NET


def get_last_epoch_net(network_path):
    model_path = max(
        [path for path in network_path.iterdir() if is_non_last_net(path)],
        key=lambda path: int(path.name.split('_')[-1].split('.')[0])
    )
    return model_path
