import json
from pathlib import Path
from typing import Dict, Tuple, Set, Callable
import numpy as np
from app.observers import Subject
from app.special_values import UNKNOWN_KEY, LAST_NET, ZERO_X_Y
import cv2


def two_points_min(point_1, point_2):
    return two_points_combine(point_1, point_2, min)


def two_points_max(point_1, point_2):
    return two_points_combine(point_1, point_2, max)


def two_points_combine(point_1, point_2, func):
    return tuple(map(lambda mn, val: func(mn, val), point_1, point_2))


def box_margin_from_box_shape(box_shape):
    box_margin = tuple(map(lambda x: x//2, box_shape))
    return box_margin


def points_accumulate(point_a, point_b, scale_b=1):
    return tuple(map(lambda axis_a, axis_b:  axis_a + axis_b * scale_b, point_a, point_b))


def points_sub(point_a, point_b):
    return points_accumulate(point_a, point_b, scale_b=-1)


def box_lines_from_center(location, box_shape):
    box_margin = box_margin_from_box_shape(box_shape)
    new_location = [axis for scale_b in [-1, 1] for axis in points_accumulate(location, box_margin, scale_b)]
    return new_location


def get_box_center(location, box_shape):
    box_margin = box_margin_from_box_shape(box_shape)
    return points_accumulate(location, box_margin)


def get_data_params_from_file(path: Path, file_name='default_config.json'):
    file_path = path / file_name
    if file_path.exists():
        with open(str(file_path)) as params_file:
            return json.load(params_file)
    else:
        return None


def get_unknown_key_image(image_shape):
    thickness = 6
    letter_margin = box_margin_from_box_shape(image_shape)
    empty_image = np.ones(tuple(reversed(image_shape))) * 255
    circle_image = cv2.circle(empty_image, letter_margin, 15, 0, thickness)
    unknown_image = cv2.line(circle_image, ZERO_X_Y, image_shape, 0, thickness)
    return unknown_image


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


def are_points_close(letter_location, location, dist):
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
