import tkinter as tk
import tkinter.ttk as tkk
from typing import Callable
from PIL import ImageTk, Image

from app.tools import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN, are_points_close


class Gui:
    def __init__(self, image_path, letters_path, save_letters_callback: Callable = None):
        self._save_letters_callback = save_letters_callback
        self._letters_path = letters_path
        self._image = Image.open(image_path)
        width, height = self._image.size

        self._instances_locations_by_letters = dict()
        self._curr_location = None

        self._window = tk.Tk()
        self._tk_image = ImageTk.PhotoImage(self._image)
        if save_letters_callback:
            self._save_button = tk.Button(self._window, text="save letters", command=self._save_letters)
            self._save_button.pack()

        self._combo = tkk.Combobox(self._window)
        self._combo['values'] = (1,)
        self._combo.current(0)
        # self._combo.grid(column=0, row=0)

        self._clear_button = tk.Button(self._window, text="clear all letters", command=self._clear_letters)
        self._clear_button.pack()

        self._canvas = tk.Canvas(self._window, width=width, height=height)
        self._canvas.pack()
        self._canvas.create_image(0, 0, image=self._tk_image, anchor=tk.NW)
        self._canvas.bind("<Button-1>", self._on_mouse_press_left)
        self._canvas.bind("<Button-3>", self._on_mouse_press_right)
        self._canvas.bind('<Motion>', self._on_mouse_motion)

    def _get_indicating_letters(self):
        return set(self._instances_locations_by_letters.keys())

    def _remove_letter(self, letter):
        self._instances_locations_by_letters.pop(letter)

    def _add_letter(self, letter):
        self._instances_locations_by_letters[letter] = set()

    def set_duplicate_letters(self, letter, locations):
        old_locations = self._instances_locations_by_letters[letter]
        self._instances_locations_by_letters[letter] = locations
        self._clear_letters()

        [self._add_a_box(location) for location in locations]

    def _clear_letters(self):
        for letter_location in self._get_indicating_letters():
            self.remove_letter(letter_location)

    def _save_letters(self):
        self._save_letters_callback(self._instances_locations_by_letters)
        print('_save_letters_callback called')

    def _add_a_box(self, letter_location):
        x_center, y_center = letter_location
        self._canvas.create_rectangle(
            x_center - BOX_WIDTH_MARGIN,
            y_center - BOX_HEIGHT_MARGIN,
            x_center + BOX_WIDTH_MARGIN,
            y_center + BOX_HEIGHT_MARGIN,
            tags=(letter_location,)
        )

    def _remove_a_box(self, letter_location):
        self._canvas.delete(letter_location)

    def add_letter(self, location):
        self._add_letter(location)
        self._add_a_box(location)

    def remove_letter(self, letter_location):
        self._remove_letter(letter_location)
        self._remove_a_box(letter_location)

    def _on_mouse_motion(self, event):
        self._curr_location = (event.x, event.y)

    def _on_mouse_press_left(self, _):
        self.add_letter(self._curr_location)

    def _on_mouse_press_right(self, _):
        location = self._curr_location
        for letter_location in self._get_indicating_letters():
            if are_points_close(letter_location, location):
                self.remove_letter(letter_location)

    def run(self):
        self._window.mainloop()


if __name__ == '__main__':
    Gui().run()
