import os
import random
import sys
import time
import json
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Union
from concurrent.futures.thread import ThreadPoolExecutor
from app.data_model import DataModel
from app.main_window import MainWindow
from app.special_values import MAX_LETTER_INCIDENTS, UNKNOWN_KEY
from app.paths import IMAGES_PATH, LETTERS_PATH, IDENTIFIER_NETS_PATH, DETECTOR_NETS_PATH, TRAIN_DATA_PATH
from app.tools import are_points_close, union_notifier_and_dict_values, union_notifier_and_dict_sets, \
    get_unknown_key_image, box_lines_from_center, get_box_center, get_data_params_from_file
from letter_classifier.identify_letter import identify_letters
from letter_classifier.train_identifier import run_train as train_identifier
from letter_detector.train_detector import run_train as train_detector
from letter_detector.find_centers import detect_letters
import logging

from tessarect.tesseract_read_page import get_tessarect_page_letters


def create_logger(logger_name=__file__):
    new_logger = logging.getLogger(logger_name)
    new_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    new_logger.addHandler(handler)
    return new_logger


logger = create_logger('main')


class App:
    def __init__(self):
        params = get_data_params_from_file(Path('./'))
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._data_model = DataModel(IMAGES_PATH)

        list_of_buttons_and_indicators = [
            (True, ('<', lambda: self._page_move(back=True))),
            (False, ('Page', self._data_model.page)),
            (True, ('>', self._page_move)),
        ]

        menu_values = {
            'Data Set': [
                ('Make all unknown (in page)', self._wrap_to_executor(self._on_all_unknown_in_page_data)),
                ('Clean all (in page)', self._wrap_to_executor(self._on_clean_page_data)),
                ('Load', self._on_load_data),
                ('Save', self._wrap_to_executor(self._on_save_data)),
            ],
            'Training Nets': [
                ('Train Detection', self._wrap_to_executor(self._run_detection_train)),
                ('Train Identification', self._wrap_to_executor(self._run_identification_train)),
                ('Train', self._wrap_to_executor(self._train_networks_last_dataset))
            ],
            'Find Letters': [
                ('Tesserct', self._wrap_to_executor(self._get_tessarect_page_letters)),
                ('Detect', self._wrap_to_executor(self._detect_letters)),
                ('Identify', self._wrap_to_executor(self._identify_letters)),
                ('Both', self._wrap_to_executor(self._run_both_nets))
            ]
        }
        self._gui = MainWindow(
            self._data_model,
            self._look_for_duplicates,
            self._get_image_patch,
            list_of_buttons_and_indicators,
            menu_values,
            page_view_shape=params['page_view_shape']
        )
        self._data_model.page.data = 0
        self._data_model.different_letters.data = {UNKNOWN_KEY: get_unknown_key_image(self._data_model.letter_shape)}

    def _wrap_to_executor(self, func):
        def func_with_exception(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.exception(e)

        def func_submit(*args, **kwargs):
            self._executor.submit(func_with_exception, *args, **kwargs)
        return func_submit

    def _get_image_patch(self, image, key):
        if key == UNKNOWN_KEY:
            return get_unknown_key_image(self._data_model.letter_shape)
        x_start, y_start, x_stop, y_stop = box_lines_from_center(key, self._data_model.letter_shape)
        letter_image = np.array(image)[y_start: y_stop, x_start: x_stop]
        return letter_image

    @staticmethod
    def _key_to_str(key: Union[Tuple, str]):
        if type(key) is tuple:
            return '_'.join([str(i) for i in key])
        else:
            return key

    @classmethod
    def _train_networks_last_dataset(cls):
        last_data_set = sorted(os.listdir('../data/annotations'))[-1]
        cls._train_networks(last_data_set)

    @staticmethod
    def _train_networks(data_set):
        train_detector(data_set)
        train_identifier(data_set)

    def _get_empty_instance_location_map(self):
        return {key: set() for key in self._data_model.different_letters.data.keys()}

    def _on_clean_page_data(self):
        self._data_model.instances_locations_by_letters.data = self._get_empty_instance_location_map()

    def _on_all_unknown_in_page_data(self):
        all_letters_in_sets = self._data_model.instances_locations_by_letters.data.values()
        all_letters = {location for locations_set in all_letters_in_sets for location in locations_set}
        empty_instances_locations_by_letters = self._get_empty_instance_location_map()
        empty_instances_locations_by_letters[UNKNOWN_KEY] = all_letters
        self._data_model.instances_locations_by_letters.data = empty_instances_locations_by_letters

    def _on_load_data(self):
        data_set_name = self._gui.get_folder(LETTERS_PATH, 'select dataset').name
        self._wrap_to_executor(lambda: self._load_data(data_set_name))()

    def _load_data(self, data_set_name):
        dataset_path = LETTERS_PATH / data_set_name

        pages_folder = dataset_path / 'pages'
        letters_folder = dataset_path / 'letters_map'

        key_image_dict = {key_path.name: cv2.imread(str(key_path)) for key_path in letters_folder.iterdir()}
        self._data_model.page._subject_state = None
        self._data_model.different_letters.data = key_image_dict

        page_folder_names = self._get_pages_names()
        for key_path in pages_folder.iterdir():
            page = page_folder_names.index(key_path.name)
            with open(str(key_path / 'instances_locations_by_index.json')) as location_dict_file:
                self._data_model.instances_locations_per_image[page] = json.load(location_dict_file)
            with open(str(key_path / 'status.json')) as status_dict_file:
                self._data_model.is_page_ready_map[page] = json.load(status_dict_file)['is_ready']

    def _on_save_data(self):
        data_set_name = time.strftime("dataset_%Y%m%d-%H%M%S")
        new_dataset_path = LETTERS_PATH / data_set_name

        letters_folder = new_dataset_path / 'letters_map'
        pages_folder = new_dataset_path / 'pages'

        ready_map = self._data_model.is_page_ready_map
        instances_locations = self._data_model.instances_locations_per_image
        page_folder_names = self._get_pages_names()

        for page_folder_name, instance_locations, is_ready in \
                zip(page_folder_names, instances_locations, ready_map):
            self._save_page(instance_locations, pages_folder, page_folder_name, is_ready)

        letters_folder.mkdir(parents=True)
        for key, image in self._data_model.different_letters.data.items():
            self.save_individual_images(image, key, letters_folder)

    def _run_identification_train(self):
        data_set_name = time.strftime("identification_dataset_%Y%m%d-%H%M%S")
        new_dataset_path = TRAIN_DATA_PATH / data_set_name

        letters_folder = new_dataset_path / 'letters_map'
        letter_images_folder = new_dataset_path / 'letter_images'

        ready_map = self._data_model.is_page_ready_map
        images_paths = self._data_model.images_paths
        instances_locations = self._data_model.instances_locations_per_image
        page_folder_names = self._get_pages_names()

        for page_path, page_folder_name, instance_locations, is_ready in \
                zip(images_paths, page_folder_names, instances_locations, ready_map):
            if is_ready:
                page_image = cv2.imread(str(page_path))
                self._save_pages_letters(page_image, instance_locations, letter_images_folder, True)

        letters_folder.mkdir(parents=True)
        for key, image in self._data_model.different_letters.data.items():
            self.save_individual_images(image, key, letters_folder)

        train_identifier(data_set_name)

    def _run_detection_train(self):
        data_set_name = time.strftime("detection_dataset_%Y%m%d-%H%M%S")
        new_dataset_path = TRAIN_DATA_PATH / data_set_name
        pages_folder = new_dataset_path / 'pages'

        ready_map = self._data_model.is_page_ready_map
        instances_locations = self._data_model.instances_locations_per_image
        page_folder_names = self._get_pages_names()

        for page_folder_name, instance_locations, is_ready in zip(page_folder_names, instances_locations, ready_map):
            if is_ready:
                self._save_page(instance_locations, pages_folder, page_folder_name, True)

        train_detector(data_set_name)

    def _get_pages_names(self):
        return [page_path.with_suffix('').name for page_path in self._data_model.images_paths]

    def _save_pages_letters(self, page: np.ndarray, instances_locations: Dict, letters_folder: Path, is_ready: bool):
        letters_folder = letters_folder if is_ready else letters_folder / 'WIP'
        for key, duplicate_letters in instances_locations.items():
            main_letter_folder = letters_folder / self._key_to_str(key)
            main_letter_folder.mkdir(parents=True, exist_ok=True)
            for letter_location in duplicate_letters:
                letter_image = self._get_image_patch(page, letter_location)
                self.save_individual_images(letter_image, letter_location, main_letter_folder)

    @staticmethod
    def _save_page(instances_locations_by_letters: Dict, pages_folder: Path, page_folder_name: str, is_ready: bool):
        path_to_save = pages_folder / page_folder_name
        path_to_save.mkdir(parents=True)
        image_to_index = {image: str(image) for i, image in enumerate(instances_locations_by_letters.keys())}

        instances_locations_by_index = {
            image_to_index[image]: list(map(lambda x: list(map(int, x)), locations))
            for image, locations in instances_locations_by_letters.items()
        }
        with open(str(path_to_save / 'instances_locations_by_index.json'), 'w') as fp:
            json.dump(instances_locations_by_index, fp, indent=4)

        with open(str(path_to_save / 'status.json'), 'w') as fp:
            json.dump({'is_ready': is_ready}, fp, indent=4)

    @classmethod
    def save_individual_images(cls, letter_image, letter_location, letters_folder):
        file_name = letters_folder / '{}.jpg'.format(cls._key_to_str(letter_location))
        cv2.imwrite(str(file_name), letter_image)

    def _run_both_nets(self):
        self._detect_letters()
        self._identify_letters()

    def _detect_letters(self):
        detector_path = max(DETECTOR_NETS_PATH.iterdir(), key=lambda x: x.name)
        logger.info('detecting letters...')
        found_locations = detect_letters(self._data_model.image_path, detector_path)
        logger.info('letters detected')
        new_values_dict = {UNKNOWN_KEY: found_locations}
        union_notifier_and_dict_sets(self._data_model.instances_locations_by_letters, new_values_dict)
        logger.info('detection done')

    def _identify_letters(self):
        identifier_path = max(IDENTIFIER_NETS_PATH.iterdir(), key=lambda x: x.name)
        logger.info('identifying letters...')
        unknown_locations = self._data_model.instances_locations_by_letters.data[UNKNOWN_KEY]
        locations_dict, letters_map = identify_letters(
            self._data_model.image_path, unknown_locations, identifier_path, self._data_model.letter_shape
        )
        logger.info('letters identified')
        union_notifier_and_dict_values(self._data_model.different_letters, letters_map)
        self._set_duplicate_letters(UNKNOWN_KEY, set())
        union_notifier_and_dict_sets(self._data_model.instances_locations_by_letters, locations_dict)
        logger.info('identification done')

    def _get_tessarect_page_letters(self):
        cv_image = np.array(self._data_model.pil_image)
        letters_and_locations = get_tessarect_page_letters(cv_image)
        dup_letters = defaultdict(set)
        for letter, location in letters_and_locations:
            dup_letters[letter].add(location)
        dup_letters = {next(iter(locations)): locations for locations in dup_letters.values()}
        key_image_map = {location: self._get_image_patch(cv_image, location) for location in dup_letters.keys()}

        union_notifier_and_dict_values(self._data_model.different_letters, key_image_map)
        union_notifier_and_dict_sets(self._data_model.instances_locations_by_letters, dict(dup_letters))

    def _look_for_duplicates(self, key, letter_center: Tuple, num_of_letters: int):
        letter_shape = self._data_model.letter_shape
        images_of_duplicated_letters = self._get_image_patch(self._data_model.pil_image, letter_center)
        nw_locations = self._basic_matching_new(np.array(self._data_model.pil_image), images_of_duplicated_letters)
        nw_locations.sort(key=lambda v: v[1], reverse=True)
        to_add = []
        to_check = nw_locations
        while to_check:
            new_point = to_check.pop(0)
            to_add.append(new_point)
            to_check = [location for location in to_check
                        if not are_points_close(location[0], new_point[0], letter_shape)]

        to_add = to_add[:num_of_letters]
        found_locations = {get_box_center(point, letter_shape) for point, val in to_add}
        self._set_duplicate_letters(key, found_locations)

    def _set_duplicate_letters(self, letter, locations):
        data = self._data_model.instances_locations_by_letters.data
        data[letter] = locations
        self._data_model.instances_locations_by_letters.data = data

    @staticmethod
    def _basic_matching_new(image, letter_patch, number_letters=MAX_LETTER_INCIDENTS):
        res = cv2.matchTemplate(image, letter_patch, cv2.TM_CCORR_NORMED)
        matching_values = np.sort(res, axis=None, kind=None, order=None)

        min_value = matching_values[-number_letters]
        above_mat = res > min_value
        locations_by_values = [
            [(x_center, y_center), res[y_center, x_center]]
            for y_center, x_center in np.transpose(np.nonzero(above_mat))
        ]
        return locations_by_values

    def _page_move(self, back=False):
        new_page = self._data_model.page.data + (-1 if back else 1)
        if 0 <= new_page < self._data_model.num_of_pages:
            self._data_model.page.data = new_page
        else:
            print('illegal page: {}'.format(new_page))

    def run(self):
        self._gui.run()


if __name__ == '__main__':
    random.seed(0)
    app = App()
    app.run()
