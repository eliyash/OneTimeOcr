import tkinter as tk
from typing import Callable

from PIL import ImageTk

from app.data_model import DataModel
from app.main_letters_handler import MainLettersHandler
from app.tools import are_points_close, NUM_OF_LETTERS


class Gui:
    def __init__(
            self,
            data_model: DataModel,
            get_image_patch: Callable,
            look_for_dups_callback: Callable,
            save_letters_callback: Callable,
            get_images_by_locations_callback: Callable
    ):
        self._data_model = data_model
        self._get_image_patch = get_image_patch
        self._look_for_dups_callback = look_for_dups_callback
        self._save_letters_callback = save_letters_callback
        self._get_images_by_locations_callback = get_images_by_locations_callback

        self._window = tk.Tk()
        self._tk_image = ImageTk.PhotoImage(self._data_model.image)
        self._top_bar = tk.Frame(self._window)
        self._top_bar.grid(row=0, column=0, sticky="nsew")

        self._save_button = tk.Button(self._top_bar, text="save lettres", command=self._on_save_all_letters)
        self._save_button.pack(side=tk.LEFT)

        self._look_for_dup_button = tk.Button(self._top_bar, text="look for duplicates", command=self._on_look_for_duplicates)
        self._look_for_dup_button.pack(side=tk.LEFT)

        self._text_frame = tk.Frame(self._window)
        self._text_frame.grid(row=1, column=0, sticky="nsew")

        self._letters_frame = tk.Frame(self._window)
        self._letters_frame.grid(row=1, column=0, sticky="nsew")

        width, height = self._data_model.image.size
        self._canvas = tk.Canvas(self._text_frame, width=width, height=height)
        self._canvas.pack(side=tk.LEFT)
        self._canvas.create_image(0, 0, image=self._tk_image, anchor=tk.NW)

        self._canvas2 = tk.Canvas(self._letters_frame, width=width, height=height)
        self._canvas2.pack(side=tk.LEFT)
        self._canvas2.create_image(0, 0, image=self._tk_image, anchor=tk.NW)

        self._text_frame.tkraise()
        # self._letters_frame.tkraise()
        self._canvas.bind("<Button-1>", self._on_mouse_press_left)
        self._canvas.bind("<Button-3>", self._on_mouse_press_right)

        self._duplicates = tk.Scale(self._top_bar, from_=1, to=200, orient=tk.HORIZONTAL)
        self._duplicates.set(NUM_OF_LETTERS)
        self._duplicates.pack(side=tk.LEFT)

        self._main_letters_handler = MainLettersHandler(self._data_model, self._top_bar, self._canvas)

    def _set_duplicate_letters(self, letter, locations):
        marker_manager, _ = self._data_model.instances_locations_by_letters[letter]
        marker_manager.set_all_letters(locations)
        self._data_model.instances_locations_by_letters[letter] = (marker_manager, locations)

    def _on_save_all_letters(self):
        self._save_letters_callback(self._data_model.instances_locations_by_letters)

    def _on_look_for_duplicates(self):
        current_main_letter = self._data_model.current_main_letter
        locations = self._look_for_dups_callback(current_main_letter, self._duplicates.get())
        self._set_duplicate_letters(current_main_letter, locations)

    def _on_mouse_press_left(self, event):
        self._main_letters_handler.add_main_letter((event.x, event.y))

    def _on_mouse_press_right(self, event):
        marker_manager, letters_locations = self._data_model.main_locations
        for letter_location in letters_locations:
            if are_points_close(letter_location, (event.x, event.y)):
                marker_manager.remove_letter(letter_location)

    def run(self):
        self._window.mainloop()
