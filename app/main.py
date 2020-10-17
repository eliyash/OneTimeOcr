from pathlib import Path

import cv2
import numpy as np
from typing import Set, Dict, Tuple
from app.gui import Gui
from app.tools import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN, are_points_close, IMAGE_PATH, LETTERS_PATH, \
    MAX_LETTER_INCIDENTS


class App:
    def __init__(self):
        self._image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE).astype('float16') / 256
        self._gui = Gui(IMAGE_PATH, self._get_image_patch, self._look_for_duplicates, self._on_save_letters)

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

    def _on_save_letters(self, letters_centers: Dict):
        folder_to_save = Path(LETTERS_PATH)
        for main_letter, (_, locations_of_duplicate_letters) in letters_centers.items():
            letters_folder = folder_to_save / '_'.join([str(i) for i in main_letter])
            letters_folder.mkdir(parents=True, exist_ok=True)
            images_of_duplicated_letters = self._get_letters_images(locations_of_duplicate_letters, self._image)
            for index, letter_image in enumerate(images_of_duplicated_letters):
                cv2.imwrite(str(letters_folder / '{}.jpg'.format(index)), (letter_image*256).astype('uint8'))

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
        found_locations = [(x_center + BOX_WIDTH_MARGIN, y_center + BOX_HEIGHT_MARGIN)
                           for (y_center, x_center), val in to_add]
        return found_locations

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
