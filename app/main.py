import time
from pathlib import Path

import json
import cv2
import numpy as np
from typing import Set, Dict, Tuple

from app.data_model import DataModel
from app.gui import Gui
from app.tools import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN, are_points_close, IMAGE_PATH, LETTERS_PATH, \
    MAX_LETTER_INCIDENTS
from letter_classifier.identify_letter import identify_letters
from letter_detector.find_centers import detect_letters


class App:
    def __init__(self):
        self._data_model = DataModel(IMAGE_PATH)
        self._image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE).astype('float16') / 256
        self._gui = Gui(
            self._data_model,
            self._get_image_patch,
            self._look_for_duplicates,
            self._network_detect,
            self._on_save_letters,
            lambda locations: self._get_letters_images(self._image, locations)
        )

    @staticmethod
    def _get_image_patch(image, key):
        (x_center, y_center) = key
        letter_image = image[
           y_center - BOX_HEIGHT_MARGIN: y_center + BOX_HEIGHT_MARGIN,
           x_center - BOX_WIDTH_MARGIN: x_center + BOX_WIDTH_MARGIN
        ]
        return letter_image

    @classmethod
    def _get_letters_images(cls, letters_centers: Set, image: np.ndarray):
        letter_images = []
        for key in letters_centers:
            letter_image = cls._get_image_patch(image, key)
            letter_images.append(letter_image)
        return letter_images

    @staticmethod
    def _location_to_str(location: Tuple):
        return '_'.join([str(i) for i in location])

    @classmethod
    def _all_locations_to_str(cls, letters_centers_dict: Dict):
        letters_centers_as_str = {
            cls._location_to_str(main_letter): [cls._location_to_str(duplicate)for duplicate in duplicates]
            for main_letter, duplicates in letters_centers_dict.items()
        }
        return letters_centers_as_str

    def _on_save_letters(self, folder, letters_centers: Dict):
        time_str = time.strftime("%Y%m%d-%H%M%S")
        folder_to_save = Path(folder if folder else LETTERS_PATH) / time_str
        folder_to_save.mkdir(exist_ok=True, parents=True)
        all_locations_to_str = self._all_locations_to_str(letters_centers)
        with open(str(folder_to_save / 'letters_centers.json'), 'w') as fp:
            json.dump(all_locations_to_str, fp)

        for main_letter, locations_of_duplicate_letters in letters_centers.items():
            letters_folder = folder_to_save / self._location_to_str(main_letter)
            letters_folder.mkdir(parents=True, exist_ok=True)
            images_of_duplicated_letters = self._get_letters_images(locations_of_duplicate_letters, self._image)
            for letter_location, letter_image in zip(locations_of_duplicate_letters, images_of_duplicated_letters):
                image_as_uint = (letter_image*256).astype('uint8')
                file_name = letters_folder / '{}.jpg'.format(self._location_to_str(letter_location))
                cv2.imwrite(str(file_name), image_as_uint)

    def _network_detect(self):
        found_locations = detect_letters(IMAGE_PATH)
        example_letter = list(found_locations)[0]
        self._data_model.main_letters.data.add(example_letter)
        self._set_duplicate_letters(example_letter, found_locations)
        letter_to_locations_dist = identify_letters(IMAGE_PATH, found_locations)
        self._data_model.main_letters.data.remove(example_letter)
        for main_letter, duplicate_letters in letter_to_locations_dist.items():
            self._set_duplicate_letters(main_letter, duplicate_letters)

    def _look_for_duplicates(self, letter_center: Tuple, num_of_letters: int):
        images_of_duplicated_letters = self._get_image_patch(self._image, letter_center)
        nw_locations = self._basic_matching_new(self._image, images_of_duplicated_letters)
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

    def run(self):
        self._gui.run()


if __name__ == '__main__':
    App().run()
