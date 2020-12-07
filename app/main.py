import random
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Union
from concurrent.futures.thread import ThreadPoolExecutor
from app.data_model import DataModel
from app.gui import Gui
from app.tools import are_points_close, BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN, MAX_LETTER_INCIDENTS, IMAGES_PATH, \
    LETTERS_PATH, UNKNOWN_KEY, UNKNOWN_IMAGE
from letter_classifier.identify_letter import identify_letters
from letter_detector.find_centers import detect_letters
import logging
logger = logging.getLogger()


class App:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._data_model = DataModel(IMAGES_PATH)
        self._gui = Gui(
            self._data_model,
            self._wrap_to_executor(self._look_for_duplicates),
            self._wrap_to_executor(self._network_detect),
            self._on_save_data,
            self._page_move,
            self._get_image_patch
        )
        self._data_model.page.data = 0
        self._data_model.different_letters.data = {UNKNOWN_KEY: UNKNOWN_IMAGE}

    def _wrap_to_executor(self, func):
        return lambda *args, **kwargs: self._executor.submit(func, *args, **kwargs)

    @staticmethod
    def _get_image_patch(image, key):
        if key == UNKNOWN_KEY:
            return UNKNOWN_IMAGE
        (x_center, y_center) = key
        letter_image = np.array(image)[
           y_center - BOX_HEIGHT_MARGIN: y_center + BOX_HEIGHT_MARGIN,
           x_center - BOX_WIDTH_MARGIN: x_center + BOX_WIDTH_MARGIN
        ]
        return letter_image

    @staticmethod
    def _key_to_str(key: Union[Tuple, str]):
        if type(key) is tuple:
            return '_'.join([str(i) for i in key])
        else:
            return key

    def _on_save_data(self):
        time_str = time.strftime("dataset_%Y%m%d-%H%M%S")
        new_dataset_path = LETTERS_PATH / time_str

        letters_folder = new_dataset_path / 'letters_map'
        pages_folder = new_dataset_path / 'pages'
        letter_images_folder = new_dataset_path / 'letter_images'

        ready_map = self._data_model.is_page_ready_map
        images_paths = self._data_model.images_paths
        instances_locations = self._data_model.instances_locations_per_image
        page_folder_names = [
            self._data_model.images_paths[page].with_suffix('').name for page in range(len(instances_locations))
        ]

        for page_path, page_folder_name, instance_locations, is_ready in \
                zip(images_paths, page_folder_names, instances_locations, ready_map):
            page_image = cv2.imread(str(page_path))
            self._save_pages_letters(page_image, instance_locations, letter_images_folder, is_ready)
            self._save_page(instance_locations, pages_folder, page_folder_name, is_ready)

        letters_folder.mkdir(parents=True)
        for key, image in self._data_model.different_letters.data.items():
            self.save_individual_images(image, key, letters_folder)

    @classmethod
    def _save_pages_letters(cls, page: np.ndarray, instances_locations: Dict, letters_folder: Path, is_ready: bool):
        letters_folder = letters_folder if is_ready else letters_folder / 'WIP'
        for key, duplicate_letters in instances_locations.items():
            main_letter_folder = letters_folder / cls._key_to_str(key)
            main_letter_folder.mkdir(parents=True, exist_ok=True)
            for letter_location in duplicate_letters:
                letter_image = cls._get_image_patch(page, letter_location)
                cls.save_individual_images(letter_image, letter_location, main_letter_folder)

    @staticmethod
    def _save_page(instances_locations_by_letters: Dict, pages_folder: Path, page_folder_name: str, is_ready: bool):
        path_to_save = (pages_folder if is_ready else pages_folder / 'WIP') / page_folder_name
        path_to_save.mkdir(parents=True)
        image_to_index = {image: str(image) for i, image in enumerate(instances_locations_by_letters.keys())}

        instances_locations_by_index = {
            image_to_index[image]: list(map(lambda x: list(map(int, x)), locations))
            for image, locations in instances_locations_by_letters.items()
        }
        with open(str(path_to_save / 'instances_locations_by_index.json'), 'w') as fp:
            json.dump(instances_locations_by_index, fp, indent=4)

    @classmethod
    def save_individual_images(cls, letter_image, letter_location, letters_folder):
        image_as_uint = letter_image
        file_name = letters_folder / '{}.jpg'.format(cls._key_to_str(letter_location))
        cv2.imwrite(str(file_name), image_as_uint)

    def _network_detect(self):
        logger.info('detecting letters')
        found_locations = detect_letters(self._data_model.image_path)
        logger.info('letters detected')
        self._set_duplicate_letters(UNKNOWN_KEY, found_locations)
        logger.info('detected letters showed, identifying letters')
        letter_to_locations_dist = identify_letters(self._data_model.image_path, found_locations)
        letter_to_locations_dist[UNKNOWN_KEY] = set()
        logger.info('letters identified')
        self._data_model.instances_locations_by_letters.data = letter_to_locations_dist
        logger.info('all identify letters showed')

    def _look_for_duplicates(self, letter_center: Tuple, num_of_letters: int):
        images_of_duplicated_letters = self._get_image_patch(self._data_model.pil_image, letter_center)
        nw_locations = self._basic_matching_new(np.array(self._data_model.pil_image), images_of_duplicated_letters)
        nw_locations.sort(key=lambda v: v[1], reverse=True)
        to_add = []
        to_check = nw_locations
        while to_check:
            new_point = to_check.pop(0)
            to_add.append(new_point)
            to_check = [location for location in to_check if not are_points_close(location[0], new_point[0])]

        to_add = to_add[:num_of_letters]
        found_locations = {(x_center + BOX_WIDTH_MARGIN, y_center + BOX_HEIGHT_MARGIN)
                           for (y_center, x_center), val in to_add}
        self._set_duplicate_letters(letter_center, found_locations)

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
            [(x_center, y_center), res[x_center, y_center]]
            for x_center, y_center in np.transpose(np.nonzero(above_mat))
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
