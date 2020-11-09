import random
import tkinter.ttk as tkk
import tkinter as tk
from typing import Dict, Tuple, Set, Callable
import numpy as np
from PIL import ImageTk, Image

from app.data_model import DataModel
from app.marker_manager import MarkerManager, SimpleMarkerDrawer
from app.tools import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN

EMPTY_IMAGE = np.ones((BOX_HEIGHT_MARGIN*2, BOX_WIDTH_MARGIN*2)) * 255


class MainLettersHandler:
    def __init__(self, data_model: DataModel, top_bar, canvas, get_image_patch: Callable):
        random.seed(0)

        self._data_model = data_model
        self._get_image_patch = get_image_patch

        # locals
        self._letters_markers_managers = dict()  # type: Dict[Tuple, SimpleMarkerDrawer]

        self._top_bar = top_bar
        self._canvas = canvas

        self._combo = tkk.Combobox(self._top_bar)
        self._combo.pack(side=tk.LEFT)
        self._combo.bind('<<ComboboxSelected>>', self._on_combo_selected)

        self._clear_button = tk.Button(self._top_bar, text="remove main letter", command=self._on_clear_letters)
        self._clear_button.pack(side=tk.LEFT)

        self._cv_image = np.array(self._data_model.image)
        self._chosen_letter_image = tk.Label(self._top_bar)
        self._chosen_letter_image.pack(side=tk.LEFT)

        self._main_markers_manager = MarkerManager(
            self._data_model.main_letters,
            self._canvas, 'black', BOX_WIDTH_MARGIN + 2, BOX_HEIGHT_MARGIN + 3
        )

        self._chosen_letter_markers_manager = MarkerManager(
            self._data_model.current_main_letter,
            self._canvas, 'green', BOX_WIDTH_MARGIN + 4, BOX_HEIGHT_MARGIN + 6
        )

        self._set_chosen_letter_image(None)
        self._data_model.instances_locations_by_letters.attach(self.set_marker_managers_for_duplicates)
        self._data_model.current_main_letter.attach(self._set_active_main_letter)
        self._data_model.current_main_letter.attach(self._set_chosen_letter_image)

    def _set_chosen_letter_image(self, letter):
        if letter:
            cv_letter_image = self._get_image_patch(self._cv_image, list(letter)[0])
        else:
            cv_letter_image = EMPTY_IMAGE
        tk_letter_image = ImageTk.PhotoImage(Image.fromarray(cv_letter_image))
        self._chosen_letter_image.config(image=tk_letter_image)
        self._chosen_letter_image.image = tk_letter_image

    @staticmethod
    def _get_random_color():
        return '#' + ''.join(['{:02x}'.format(random.randint(0, 255)) for _ in range(3)])

    def _on_clear_letters(self):
        self._remove_main_letter(self._data_model.current_main_letter.data)

    def add_main_letter(self, letter_location):
        data = self._data_model.main_letters.data
        data.add(letter_location)
        self._data_model.main_letters.data = data
        self._data_model.current_main_letter.data = {letter_location}

    def _remove_main_letter(self, letter):
        main_letters = self._data_model.main_letters.data  # type: Set
        main_letters -= letter
        self._data_model.current_main_letter.data = {list(main_letters)[0]} if main_letters else set()
        self._data_model.main_letters.data = main_letters

    def _set_active_main_letter(self, letter):
        if letter:
            self._set_combo(list(letter)[0])
        else:
            self._reset_combo()

    def _reset_combo(self):
        self._combo['values'] = (' ',)
        self._combo.current(0)

    def _set_combo(self, letter):
        values = tuple(self._data_model.main_letters.data)
        index = values.index(letter)
        self._combo['values'] = values
        self._combo.current(index)

    def _on_combo_selected(self, _):
        x, y = (int(val_as_string) for val_as_string in self._combo.get().split(' '))
        self._data_model.current_main_letter.data = {(x, y)}

    def handle_change_in_main_letters(self, new_main_letters):
        old_main_letter = set(self._letters_markers_managers.keys())
        letters_to_add = new_main_letters - old_main_letter
        letters_to_remove = old_main_letter - new_main_letters
        [self._letters_markers_managers.pop(letter_to_remove) for letter_to_remove in letters_to_remove]
        for letter_to_add in letters_to_add:
            self._letters_markers_managers[letter_to_add] = SimpleMarkerDrawer(self._canvas, self._get_random_color())

    def set_marker_managers_for_duplicates(self, locations_dict):
        self.handle_change_in_main_letters(set(locations_dict.keys()))
        for main_letter in locations_dict.keys():
            self._letters_markers_managers[main_letter].update_letters(locations_dict[main_letter])
