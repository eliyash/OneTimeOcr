from typing import Dict

from PIL import Image


class DataModel:
    def __init__(self, image_path: str):
        self.image = Image.open(image_path)
        self.current_main_letter = None
        self.instances_locations_by_letters = dict()  # type: Dict[Dict]
        self.image_by_letters = dict()  # type: Dict

    @property
    def main_locations(self):
        return set(self.instances_locations_by_letters.keys())

    @property
    def current_location_duplicates(self):
        if self.instances_locations_by_letters:
            return self.instances_locations_by_letters[self.current_main_letter]
        else:
            return []
