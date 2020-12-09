import random
from typing import Dict, Tuple, Callable

from app.data_model import ViewModel
from app.marker_drawer import SimpleMarkerDrawer
from app.special_values import UNKNOWN_KEY
from app.tools import get_values_to_add_and_remove


class LettersInPageHandler:
    def __init__(
            self, view_model: ViewModel, run_gui_action: Callable,
            canvas, get_image_patch: Callable, translator: Callable
    ):
        self._view_model = view_model

        self._canvas = canvas
        self._get_image_patch = get_image_patch
        self._translator = translator

        # locals
        self._letters_markers_managers = dict()  # type: Dict[Tuple, SimpleMarkerDrawer]

        self._view_model.data_model.instances_locations_by_letters.attach(
            run_gui_action(self.set_marker_managers_for_duplicates)
        )

    @staticmethod
    def _get_random_color():
        return '#' + ''.join(['{:02x}'.format(random.randint(0, 255)) for _ in range(3)])

    def add_main_letter(self, location):
        data = self._view_model.data_model.different_letters.data
        data[location] = self._get_image_patch(location)
        self._view_model.data_model.different_letters.data = data
        if location != UNKNOWN_KEY:
            self.add_dup_letter(location, location)

    def add_dup_letter(self, key, location):
        data = self._view_model.data_model.instances_locations_by_letters.data
        data[key].add(location)
        self._view_model.data_model.instances_locations_by_letters.data = data

    def handle_change_in_main_letters(self, new_main_letters):
        to_remove, to_add = get_values_to_add_and_remove(self._letters_markers_managers, new_main_letters)
        [self._letters_markers_managers.pop(letter_to_remove) for letter_to_remove in to_remove]
        for letter_to_add in to_add:
            marker_drawer = SimpleMarkerDrawer(self._canvas, self._get_random_color(), translator=self._translator)
            self._letters_markers_managers[letter_to_add] = marker_drawer

    def set_marker_managers_for_duplicates(self, locations_dict):
        self.handle_change_in_main_letters(locations_dict)
        for main_letter in locations_dict.keys():
            self._letters_markers_managers[main_letter].update_letters(locations_dict[main_letter])
