import random
from typing import Dict

from PIL import Image

from app.observers import Subject


class DataModel:
    def __init__(self, image_path: str):
        self._image = Image.open(image_path)
        self.instances_locations_by_letters = Subject(dict())

    @property
    def image(self):
        return self._image

    def reset_data(self):
        random.seed(0)
        current_data = self.instances_locations_by_letters.data
        self.instances_locations_by_letters.data = dict()
        self.instances_locations_by_letters.data = current_data


class ViewModel:
    def __init__(self, data_modal: DataModel):
        self.data_model = data_modal

        self.current_main_letters = Subject(set())
        self.current_chosen_letter = Subject()
        self.current_location_duplicates = Subject(set())

        self.current_chosen_letter.attach(self.set_new_chosen_letter)
        self.data_model.instances_locations_by_letters.attach(self.handle_main_letters_change)
        self.data_model.instances_locations_by_letters.attach(self.set_current_location_duplicates)

        self.map_keys_by_widgets = {}

    def handle_main_letters_change(self, new_instances_locations_by_letters: Dict):
        current_chosen_letter = self.current_chosen_letter.data
        new_main_letters = set(new_instances_locations_by_letters.keys())
        letters_to_add = new_main_letters - self.current_main_letters.data
        if letters_to_add:
            self.current_chosen_letter.data = next(iter(letters_to_add))
        elif current_chosen_letter in new_main_letters:
            pass
        elif new_main_letters:
            self.current_chosen_letter.data = next(iter(new_main_letters))
        else:
            self.current_chosen_letter.data = None

        self.current_main_letters.data = new_main_letters

    def set_current_location_duplicates(self, new_instances_locations_by_letters: Dict):
        main_letter = self.current_chosen_letter.data
        if main_letter in new_instances_locations_by_letters:
            new_current_location_duplicates = new_instances_locations_by_letters[main_letter]
            if self.current_location_duplicates.data != new_current_location_duplicates:
                self.current_location_duplicates.data = new_current_location_duplicates

    def set_new_chosen_letter(self, new_main_letter):
        instances_locations_by_letters = self.data_model.instances_locations_by_letters.data
        if new_main_letter in instances_locations_by_letters:
            new_current_location_duplicates = instances_locations_by_letters[new_main_letter]
        else:
            new_current_location_duplicates = []
        self.current_location_duplicates.data = new_current_location_duplicates
