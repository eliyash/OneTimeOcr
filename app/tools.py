from pathlib import Path
from typing import Dict, Tuple, Set, Callable

import numpy as np

from app.observers import Subject
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


def union_dicts_dest_src(dict_dest: Dict, dict_src: Dict, combiner: Callable):
    for key, set_value in dict_src.items():
        dict_dest[key] = combiner(dict_dest[key] if key in dict_dest else None, dict_src[key])


def union_notifier_and_dict(notifier: Subject, dict_src: Dict, cell_combiner: Callable):
    dict_dest = notifier.data
    union_dicts_dest_src(dict_dest, dict_src, cell_combiner)
    notifier.data = dict_dest


def union_notifier_and_dict_sets(notifier: Subject, dict_src: Dict):
    def union_sets(dest_set, src_set):
        dest_set = set() if dest_set is None else dest_set
        return dest_set.union(src_set)
    union_notifier_and_dict(notifier, dict_src, union_sets)


def union_notifier_and_dict_values(notifier: Subject, dict_src: Dict):
    def union_values(dest_val, src_val):
        return src_val if dest_val is None else dest_val
    union_notifier_and_dict(notifier, dict_src, union_values)
