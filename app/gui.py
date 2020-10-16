import random
import tkinter as tk
import tkinter.ttk as tkk
from typing import Callable, Dict
from PIL import ImageTk, Image

from app.colors import COLORS
from app.marker_manager import MarkerManager
from app.tools import are_points_close, BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN


class Gui:
    def __init__(self, image_path, letters_path, look_for_dups_callback: Callable, save_letters_callback: Callable):
        self._look_for_dups_callback = look_for_dups_callback
        self._save_letters_callback = save_letters_callback
        self._letters_path = letters_path
        self._image = Image.open(image_path)
        self._current_location = None
        self._current_main_letter = None
        self._instances_locations_by_letters = dict()  # type: Dict[Dict]

        self._window = tk.Tk()
        self._tk_image = ImageTk.PhotoImage(self._image)
        self._top_bar = tk.Frame(self._window)
        self._top_bar.pack(side=tk.TOP)

        self._save_button = tk.Button(self._top_bar, text="save lettres", command=self._on_save_all_letters)
        self._save_button.pack(side=tk.LEFT)

        self._look_for_dup_button = tk.Button(self._top_bar, text="look for duplicates", command=self._on_look_for_duplicates)
        self._look_for_dup_button.pack(side=tk.LEFT)

        self._clear_button = tk.Button(self._top_bar, text="clear chosen main letter", command=self._on_clear_letters)
        self._clear_button.pack(side=tk.LEFT)

        self._combo = tkk.Combobox(self._top_bar)
        self._combo.pack(side=tk.LEFT)
        self._combo.bind('<<ComboboxSelected>>', self._on_combo_selected)

        width, height = self._image.size
        self._canvas = tk.Canvas(self._window, width=width, height=height)
        self._canvas.pack(side=tk.BOTTOM)
        self._canvas.create_image(0, 0, image=self._tk_image, anchor=tk.NW)

        self._canvas.bind("<Button-1>", self._on_mouse_press_left)
        self._canvas.bind("<Button-3>", self._on_mouse_press_right)
        self._canvas.bind('<Motion>', self._on_mouse_motion)

        self._main_markers_manager = MarkerManager(self._canvas, 'black', BOX_WIDTH_MARGIN + 2, BOX_HEIGHT_MARGIN + 3)

    def _reset_combo(self):
        self._current_main_letter = None
        self._combo['values'] = (' ',)
        self._combo.current(0)

    def _set_combo(self, letter):
        self._current_main_letter = letter if letter else list(self._instances_locations_by_letters.keys())[0]
        self._combo['values'] = tuple(self._instances_locations_by_letters.keys())
        self._combo.current(self._combo['values'].index(self._current_main_letter))

    def _set_active_main_letter(self, letter):
        if self._instances_locations_by_letters:
            self._set_combo(letter)
        else:
            self._reset_combo()

    def _get_indicating_letters(self):
        return set(self._instances_locations_by_letters.keys())

    def _remove_main_letter(self, letter):
        marker_manager, _ = self._instances_locations_by_letters.pop(letter)
        marker_manager.remove_all_letters()
        self._main_markers_manager.remove_letter(letter)
        self._set_active_main_letter(None)

    def _add_main_letter(self, letter_location):
        marker_manager = MarkerManager(self._canvas, COLORS[random.randint(0, len(COLORS)-1)])
        self._instances_locations_by_letters[letter_location] = (marker_manager, set())
        self._main_markers_manager.add_letter(letter_location)
        self._set_active_main_letter(letter_location)

    def _on_save_all_letters(self):
        self._remove_main_letter(self._current_main_letter)

    def _on_clear_letters(self):
        self._remove_main_letter(self._current_main_letter)

    def _on_look_for_duplicates(self):
        self._look_for_dups_callback(set(self._instances_locations_by_letters.keys()))

    def _on_mouse_motion(self, event):
        self._current_location = (event.x, event.y)

    def _on_mouse_press_left(self, _):
        self._add_main_letter(self._current_location)

    def _on_mouse_press_right(self, _):
        location = self._current_location
        marker_manager, letters_locations = self._instances_locations_by_letters[self._current_main_letter]
        for letter_location in letters_locations:
            if are_points_close(letter_location, location):
                marker_manager.remove_letter(letter_location)

    def _on_combo_selected(self, _):
        x, y = (int(val_as_string) for val_as_string in self._combo.get().split(' '))
        self._set_active_main_letter((x, y))

    def set_duplicate_letters(self, letter, locations):
        marker_manager, _ = self._instances_locations_by_letters[letter]
        marker_manager.set_all_letters(locations)
        self._instances_locations_by_letters[letter] = (marker_manager, locations)

    def run(self):
        self._window.mainloop()
