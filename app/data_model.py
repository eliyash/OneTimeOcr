import random
import numpy as np
from PIL import Image
from typing import Dict, List

from app.observers import Subject


class DataModel:
    def __init__(self, image_paths: List[str]):
        self._images_paths = image_paths
        self._instances_locations_per_image = [dict() for _ in image_paths]
        self._is_page_ready_map = [False for _ in image_paths]

        self.instances_locations_by_letters = Subject(dict())

        self.page = Subject(None)
        self.page.attach(self.set_page)
        self.current_page = None

        self.pil_image = None
        self.cv_image = None
        self.image_path = None
        self.page.data = 0

    @property
    def num_of_pages(self):
        return len(self._images_paths)

    def set_page_state(self, value):
        self._is_page_ready_map[self.current_page] = value

    def set_page(self, index: int):
        if self.current_page:
            self._instances_locations_per_image[self.current_page] = self.instances_locations_by_letters.data
        self.image_path = self._images_paths[index]
        self.pil_image = Image.open(str(self.image_path))
        self.cv_image = np.array(self.pil_image)
        self.instances_locations_by_letters.data = self._instances_locations_per_image[index]
        self.current_page = index

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
