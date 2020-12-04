import random
import tkinter as tk
from typing import Dict, Tuple, Callable
import numpy as np
from PIL import ImageTk, Image

from app.data_model import ViewModel
from app.marker_manager import MarkerManager, SimpleMarkerDrawer
from app.observers import Subject
from app.tools import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN

EMPTY_IMAGE = np.ones((BOX_HEIGHT_MARGIN*2, BOX_WIDTH_MARGIN*2)) * 255


class MainLettersHandler:
    def __init__(
            self, view_model: ViewModel, run_gui_action: Callable,
            top_bar, canvas, get_image_patch: Callable, translator: Callable
    ):
        self._view_model = view_model
        self._run_gui_action = run_gui_action
        self._get_image_patch = get_image_patch
        self._translator = translator

        # locals
        self._old_main_letters = set()
        self._letters_markers_managers = dict()  # type: Dict[Tuple, SimpleMarkerDrawer]
        self._current_main_letter_as_set = Subject(set())

        self._top_bar = top_bar
        self._canvas = canvas

        self._cv_image = np.array(self._view_model.data_model.image)
        self._chosen_letter_image = tk.Label(self._top_bar)
        self._chosen_letter_image.pack(side=tk.LEFT)

        self._chosen_letter_markers_manager = MarkerManager(
            self._current_main_letter_as_set, run_gui_action,
            self._canvas, 'green', BOX_WIDTH_MARGIN + 4, BOX_HEIGHT_MARGIN + 6, translator=self._translator
        )

        self._set_chosen_letter_image(None)
        self._view_model.data_model.instances_locations_by_letters.attach(
            run_gui_action(self.set_marker_managers_for_duplicates)
        )
        self._view_model.current_chosen_letter.attach(self._on_current_main_letter)
        self._view_model.current_chosen_letter.attach(run_gui_action(self._set_chosen_letter_image))

    def _on_current_main_letter(self, letter):
        self._current_main_letter_as_set.data = {letter} if letter else set()

    def _set_chosen_letter_image(self, letter):
        if letter:
            cv_letter_image = self._get_image_patch(self._cv_image, letter)
        else:
            cv_letter_image = EMPTY_IMAGE
        tk_letter_image = ImageTk.PhotoImage(Image.fromarray(cv_letter_image))
        self._chosen_letter_image.config(image=tk_letter_image)
        self._chosen_letter_image.image = tk_letter_image

    @staticmethod
    def _get_random_color():
        return '#' + ''.join(['{:02x}'.format(random.randint(0, 255)) for _ in range(3)])

    def add_main_letter(self, letter_location):
        data = self._view_model.data_model.instances_locations_by_letters.data
        data[letter_location] = {letter_location}
        self._view_model.data_model.instances_locations_by_letters.data = data

    def handle_change_in_main_letters(self, new_main_letters):
        old_main_letter = set(self._letters_markers_managers.keys())
        letters_to_add = new_main_letters - old_main_letter
        letters_to_remove = old_main_letter - new_main_letters
        [self._letters_markers_managers.pop(letter_to_remove) for letter_to_remove in letters_to_remove]
        for letter_to_add in letters_to_add:
            marker_drawer = SimpleMarkerDrawer(self._canvas, self._get_random_color(), translator=self._translator)
            self._letters_markers_managers[letter_to_add] = marker_drawer

    def set_marker_managers_for_duplicates(self, locations_dict):
        self.handle_change_in_main_letters(set(locations_dict.keys()))
        for main_letter in locations_dict.keys():
            self._letters_markers_managers[main_letter].update_letters(locations_dict[main_letter])
