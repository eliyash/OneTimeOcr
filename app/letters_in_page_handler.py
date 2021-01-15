import random
from typing import Dict, Tuple, Callable

from app.data_model import ViewModel
from app.marker_drawer import SimpleMarkerDrawer
from app.special_values import UNKNOWN_KEY
from app.tools import get_values_to_add_and_remove


class LettersInPageHandler:
    def __init__(
            self, view_model: ViewModel, run_on_new_gui_thread: Callable,
            canvas, get_image_patch: Callable, translator: Callable
    ):
        self._view_model = view_model

        self._canvas = canvas
        self._get_image_patch = get_image_patch
        self._translator = translator

        # locals
        self._letters_markers_managers = dict()  # type: Dict[Tuple, SimpleMarkerDrawer]

        self._view_model.data_model.instances_locations_by_letters.attach(
            run_on_new_gui_thread(self.set_marker_managers_for_duplicates)
        )

    @staticmethod
    def _get_random_color():
        return '#' + ''.join(['{:02x}'.format(random.randint(0, 255)) for _ in range(3)])

    def add_main_letter(self, location):
        image = self._get_image_patch(location)
        if location == UNKNOWN_KEY:
            key_name = UNKNOWN_KEY
        else:
            key_name = str(location)
        data = self._view_model.data_model.different_letters.data
        data[key_name] = image
        self._view_model.data_model.different_letters.data = data
        if key_name != UNKNOWN_KEY:
            self.add_dup_letter(key_name, location)

    def add_dup_letter(self, key, location):
        data = self._view_model.data_model.instances_locations_by_letters.data
        data[key].add(location)
        self._view_model.data_model.instances_locations_by_letters.data = data

    def handle_change_in_main_letters(self, new_main_letters):
        letter_shape = self._view_model.data_model.letter_shape
        to_remove, to_add = get_values_to_add_and_remove(self._letters_markers_managers, new_main_letters)
        [self._letters_markers_managers.pop(letter_to_remove) for letter_to_remove in to_remove]
        for letter_to_add in to_add:
            marker_drawer = SimpleMarkerDrawer(
                self._view_model, self._canvas, self._get_random_color(), letter_shape, self._translator
            )
            self._letters_markers_managers[letter_to_add] = marker_drawer

    def set_marker_managers_for_duplicates(self, locations_dict):
        self.handle_change_in_main_letters(locations_dict)
        for main_letter in locations_dict.keys():
            self._letters_markers_managers[main_letter].update_letters(locations_dict[main_letter])
