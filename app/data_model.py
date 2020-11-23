from typing import Dict

from PIL import Image

from app.observers import Subject


class DataModel:
    def __init__(self, image_path: str):
        self._image = Image.open(image_path)

        self.main_letters = Subject(set())
        self.instances_locations_by_letters = Subject(dict())
        self.current_main_letter = Subject()
        self.current_location_duplicates = Subject(set())

        self.main_letters.attach(self.handle_main_letters_change)
        self.current_main_letter.attach(self.set_current_location_duplicates)
        self.instances_locations_by_letters.attach(self.set_current_location_duplicates)

    @property
    def image(self):
        return self._image

    def handle_main_letters_change(self, new_main_letters):
        map_letters = self.instances_locations_by_letters.data  # type: Dict
        current_main_letters = set(map_letters.keys())
        letters_to_add = new_main_letters - current_main_letters
        letters_to_remove = current_main_letters - new_main_letters
        [map_letters.pop(letter_to_remove) for letter_to_remove in letters_to_remove]
        [map_letters.update({letter_to_add: set()}) for letter_to_add in letters_to_add]
        self.current_main_letter.data = set()
        self.instances_locations_by_letters.data = map_letters
        if new_main_letters:
            self.current_main_letter.data = {list(letters_to_add)[0]}
        elif map_letters:
            self.current_main_letter.data = {list(new_main_letters.keys())[0]}

    def set_current_location_duplicates(self, _):
        new_main_letter = self.current_main_letter.data
        if new_main_letter:
            location_new_letter = list(new_main_letter)[0]
            new_current_location_duplicates = self.instances_locations_by_letters.data[location_new_letter]
        else:
            new_current_location_duplicates = []
        self.current_location_duplicates.data = new_current_location_duplicates
