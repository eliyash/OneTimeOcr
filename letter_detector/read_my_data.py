import json

import cv2
import numpy as np

from app.tools import box_lines_from_center

NUM_H_W = (10, 10)
IMAGE_TO_SHOW = 10
IMAGE_PATH = r"C:\Workspace\MyOCR\EAST\test_pages\test_gez good - Copy.jpg"
DATA_PATH = r"C:\Workspace\MyOCR\EAST\test_pages\test_gez good - Copy\20201116-133554\letters_centers.json"


def extract_vertices(lines):
    labels = []
    vertices = []
    for line in lines:
        vertices.append(list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
        label = 0 if '###' in line else 1
        labels.append(label)
    return np.array(vertices), np.array(labels)


def load_data(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


def name_to_location(name):
    values = name.split('_')
    values = map(int, values)
    return tuple(values)


DIRECTIONS = [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]


def center_to_all_points(point, letter_shape):
    x_start, x_stop, y_start, y_stop = box_lines_from_center(point, letter_shape)
    all_points = [val for y in [y_start, y_stop] for x in [x_start, x_stop] for val in (x, y)]
    return all_points


def create_data(path):
    centers_by_main_letters = load_data(path)
    all_centers = [center for letter_centers in centers_by_main_letters.values() for center in letter_centers]
    return np.array(all_centers)


def change_image(data, image):
    for line in data:
        line = [int(num_as_str) for num_as_str in line[:-1]]
        locations = list(zip(line[0::2], line[1::2]))
        c = 0
        for p1, p2 in zip(locations, locations[1:] + [locations[0]]):
            image = cv2.line(image, p1, p2, (100, c, 0))
            c += 50
    return image


def main():
    data = create_data(DATA_PATH, (50, 50))[0]
    path = r"C:\Workspace\MyOCR\EAST\eli east gez\test\letters\asd.txt"
    with open(path, 'w') as csv_file:
        [csv_file.write(','.join(map(str, line)) + '\n') for line in data]

    image = cv2.imread(IMAGE_PATH)
    image = change_image(data, image)
    cv2.imshow('asd', image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
