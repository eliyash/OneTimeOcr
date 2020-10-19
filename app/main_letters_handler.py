import random
import tkinter.ttk as tkk
import tkinter as tk
from app.marker_manager import MarkerManager
from app.tools import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN


class MainLettersHandler:
    def __init__(self, data_model, top_bar, canvas):
        random.seed(0)
        self._data_model = data_model
        self._top_bar = top_bar
        self._canvas = canvas
        self._instances_locations = set()
        self._combo = tkk.Combobox(self._top_bar)
        self._combo.pack(side=tk.LEFT)
        self._combo.bind('<<ComboboxSelected>>', self._on_combo_selected)

        self._clear_button = tk.Button(self._top_bar, text="remove main letter", command=self._on_clear_letters)
        self._clear_button.pack(side=tk.LEFT)

        self._main_markers_manager = MarkerManager(self._canvas, 'black', BOX_WIDTH_MARGIN + 2, BOX_HEIGHT_MARGIN + 3)
        self._chosen_letter_markers_manager = MarkerManager(self._canvas, 'green', BOX_WIDTH_MARGIN + 4, BOX_HEIGHT_MARGIN + 6)

    @staticmethod
    def _get_random_color():
        return '#' + ''.join(['{:02x}'.format(random.randint(0, 255)) for _ in range(3)])

    def _on_clear_letters(self):
        self._remove_main_letter(self._data_model.current_main_letter)

    def add_main_letter(self, letter_location):
        marker_manager = MarkerManager(self._canvas, self._get_random_color())
        self._data_model.instances_locations_by_letters[letter_location] = (marker_manager, set())
        self._main_markers_manager.add_letter(letter_location)
        self._set_active_main_letter(letter_location)

    def _remove_main_letter(self, letter):
        marker_manager, _ = self._data_model.instances_locations_by_letters.pop(letter)
        marker_manager.remove_all_letters()
        self._main_markers_manager.remove_letter(letter)
        self._set_active_main_letter(None)

    def _set_active_main_letter(self, letter):
        if self._data_model.instances_locations_by_letters:
            self._set_combo(letter)
        else:
            self._reset_combo()

    def _reset_combo(self):
        self._data_model.current_main_letter = None
        self._combo['values'] = (' ',)
        self._combo.current(0)
        self._chosen_letter_markers_manager.remove_all_letters()

    def _set_combo(self, letter):
        self._data_model.current_main_letter = letter if letter else list(self._data_model.instances_locations_by_letters.keys())[0]
        self._combo['values'] = tuple(self._data_model.instances_locations_by_letters.keys())
        self._combo.current(self._combo['values'].index(self._data_model.current_main_letter))
        self._chosen_letter_markers_manager.remove_all_letters()
        self._chosen_letter_markers_manager.add_letter(self._data_model.current_main_letter)

    def _on_combo_selected(self, _):
        x, y = (int(val_as_string) for val_as_string in self._combo.get().split(' '))
        self._set_active_main_letter((x, y))
