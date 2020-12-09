import json

import cv2
import numpy as np

from app.tools import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN

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


def center_to_all_points(point):
    x, y = point
    all_points_tuples = [(x + x_d*BOX_WIDTH_MARGIN, y + y_d*BOX_HEIGHT_MARGIN)
                         for x_d, y_d in DIRECTIONS]
    all_points = []
    for x_p, y_p in all_points_tuples:
        all_points.append(x_p)
        all_points.append(y_p)
    return all_points


def create_data(path):
    centers_by_main_letters = load_data(path)
    all_centers = [center for letter_centers in centers_by_main_letters.values() for center in letter_centers]
    vertices = list(map(center_to_all_points, all_centers))
    labels = list(map(lambda x: 1, vertices))
    return np.array(vertices), np.array(labels)


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
    data = create_data(DATA_PATH)[0]
    path = r"C:\Workspace\MyOCR\EAST\eli east gez\test\letters\asd.txt"
    with open(path, 'w') as csv_file:
        [csv_file.write(','.join(map(str, line)) + '\n') for line in data]

    image = cv2.imread(IMAGE_PATH)
    image = change_image(data, image)
    cv2.imshow('asd', image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
