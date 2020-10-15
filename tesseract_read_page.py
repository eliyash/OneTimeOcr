import os
from typing import List

import cv2
import pytesseract
import numpy as np
from pytesseract import Output

from utils import make_dir
from paths import Locations

pytesseract.pytesseract.tesseract_cmd = Locations.TESSERACT_EXEC
LETTERS_PATCH_FOLDER = r'C:\Workspace\MyOCR\identifieng letters\data\books\letters'


class Square:
    def __init__(self, top: int, height: int, left: int, width: int):
        self.top = top
        self.height = height
        self.left = left
        self.width = width

    def get_surrounding_line(self):
        top, height, left, width = self.top, self.height, self.left, self.width
        return [(left, top), (left, top + height), (left + width, top + height), (left + width, top), (left, top)]

    def get_image_cropping(self):
        top, height, left, width = self.top, self.height, self.left, self.width
        return left, top, left + width, top + height

    def clone(self):
        return Square(self.top, self.height, self.left, self.width)

    def match_top_and_height(self, match_to: "Square"):
        self.top = match_to.top
        self.height = match_to.height


class Letter:
    def __init__(self, name: str, bounding_box: Square):
        self.name = name
        self.bounding_box = bounding_box

    def clone(self):
        return Letter(self.name, self.bounding_box.clone())


class Word:
    def __init__(self, line: int, word_str: str, bounding_box: Square, letters: List[Letter] = None):
        self.line = line
        self.word_str = word_str
        self.bounding_box = bounding_box
        self.letters = letters
        if self.letters is None:
            self.letters = []

    def match_letters_to_word_boxes(self):
        for letter in self.letters:
            letter.bounding_box.match_top_and_height(self.bounding_box)

    def clone(self):
        return Word(self.line, self.word_str, self.bounding_box.clone(), [letter.clone() for letter in self.letters])


def generate_word_boxes(img):
    word_data = pytesseract.image_to_data(img, output_type=Output.DICT, lang='heb')
    # from utils import save_data, load_data
    # word_data = load_data(Locations.TESSERACT_RESULT_FOLDER + "word_data.json")
    # save_data(word_data, Locations.TESSERACT_RESULT_FOLDER + "word_data.json")

    boxes_from_tesseract = zip(
        word_data['top'],
        word_data['height'],
        word_data['left'],
        word_data['width'],
        word_data['line_num'],
        word_data['level'],
        word_data['text']
    )

    all_words = []
    line_words = []
    for top, height, left, width, line, level, text in boxes_from_tesseract:
        if level == 5:
            line_words.append(Word(line, text, Square(top, height, left, width)))
        if level == 4:
            line_words.reverse()
            all_words.extend(line_words)
            line_words = []
    line_words.reverse()
    all_words.extend(line_words)
    return all_words


def generate_letter_boxes(img):
    letter_data = pytesseract.image_to_boxes(img, output_type=Output.DICT, lang='heb')
    # from utils import save_data, load_data
    # letter_data = load_data(Locations.TESSERACT_RESULT_FOLDER + "letter_data.json")
    # save_data(letter_data, Locations.TESSERACT_RESULT_FOLDER + "letter_data.json")

    img_width, img_height = img.shape[:2]

    letters_boxes = zip(
        letter_data['char'],
        letter_data['top'],
        letter_data['bottom'],
        letter_data['left'],
        letter_data['right']
    )

    boxes_of_letters = []
    for char, top, bottom, left, right in letters_boxes:
        height = top - bottom
        top = img_height - top
        left = left
        width = right - left
        boxes_of_letters.append(Letter(char, Square(top, height, left, width)))
    return boxes_of_letters


def save_letter_images(img, letters: List[Letter]):
    index = 0
    for letter in letters:
        char = letter.name
        try:
            letter_image = img.copy()
            left, top, right, bottom = letter.bounding_box.get_image_cropping()
            letter_image = letter_image[left:right, top:bottom]
            if not char.isalpha():
                char = ord(char)
            file_path = r"{}\letter{}\\".format(Locations.PAGE_LETTERS_DIRECTORY, char)
            make_dir(file_path)
            letter_image.save('{}{}.png'.format(file_path, index), "PNG")

            index += 1
        except Exception as e:
            print("error in letter: {}, e={}".format(char, e))


def match_letters_with_words(words: List[Word], letters: List[Letter]):
    last_letters = 0
    for word in words:
        last_letter_in_current_word = last_letters + len(word.word_str)
        word.letters = letters[last_letters:last_letter_in_current_word]
        last_letters = last_letter_in_current_word


def index_to_color(index):
    r = (index * 60) % 255
    g = (index * 30) % 255
    b = (index * 15) % 255
    return r, g, b, 255


def cv_lines_draw(img, points, color, thickness=None, line_type=None, shift=None):
    new_image = img.copy()
    for index in range(len(points)-1):
        new_image = cv2.line(
            new_image, points[index], points[index+1], color=color, thickness=thickness, lineType=line_type, shift=shift
        )
    return new_image


def generate_image_with_boxes(orig_image: np.ndarray, words: List[Word]):
    new_image = orig_image.copy()
    index = 0
    word_letters = []
    for word in words:
        for letter in word.letters:
            color = (0, 0, 0, 255)
            if index:
                color = index_to_color(index)

                word_letter = letter.clone()

                new_image = cv_lines_draw(new_image, word_letter.bounding_box.get_surrounding_line(), color=color)
                word_letters.append(word_letter)

            new_image = cv_lines_draw(new_image, word.bounding_box.get_surrounding_line(), color=color)
        index += 1
    return new_image


def generate_image_with_word_boxes(orig_image: np.ndarray, words: List[Word]):
    new_image = orig_image.copy()
    for word in words:
        color = (0, 0, 0, 255)
        new_image = cv_lines_draw(new_image, word.bounding_box.get_surrounding_line(), color=color)
    return new_image


def fix_holes_and_remove_noise(image):
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=2)
    image = cv2.dilate(image, kernel, iterations=2)
    return image


def basic_matching():
    image = cv2.imread(Locations.PAGE_TO_READ_PATH, cv2.IMREAD_GRAYSCALE).astype('float16')/256
    letters_locations = dict()
    letter_locations_image = np.zeros(shape=(1520, 1520))
    for letter in os.listdir(LETTERS_PATCH_FOLDER):
        letter_path = os.path.join(LETTERS_PATCH_FOLDER, letter)
        letter_patch = cv2.imread(letter_path, cv2.IMREAD_GRAYSCALE).astype('float16')/256
        # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        #             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
        res = cv2.matchTemplate(image, letter_patch, cv2.TM_CCORR_NORMED)
        res[res > 0.94] = 1
        res[res < 1] = 0
        letter_locations_image += res[:1520, :1520]
        letters_locations[letter] = np.nonzero(res)

    cv2.imshow('p', letter_locations_image)
    cv2.imshow('image', image)
    cv2.waitKey()


def main():
    orig_image = cv2.imread(Locations.PAGE_TO_READ_PATH)

    letters = generate_letter_boxes(orig_image)
    words = generate_word_boxes(orig_image)

    match_letters_with_words(words, letters)

    for word in words:
        word.match_letters_to_word_boxes()

    orig_image = generate_image_with_boxes(orig_image, words)
    # orig_image = generate_image_with_word_boxes(orig_image, words)
    orig_image = cv2.imshow('asd', orig_image)
    cv2.waitKey()

    save_letter_images(orig_image, [letter for word in words for letter in word.letters])


if __name__ == '__main__':
    main()
    # basic_matching()
