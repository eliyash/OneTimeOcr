import csv
import cv2


IMAGE_TO_SHOW = 10
IMAGE_PATH = r"C:\Users\Eliahu\Downloads\ch4_test_images\img_{}.jpg".format(IMAGE_TO_SHOW)
DATA_PATH = r"C:\Users\Eliahu\Downloads\Challenge4_Test_Task1_GT\gt_img_{}.txt".format(IMAGE_TO_SHOW)


def draw_box_on_image(image, line):
    locations = list(zip(line[0::2], line[1::2]))
    c = 0
    for p1, p2 in zip(locations, locations[1:] + [locations[0]]):
        image = cv2.line(image, p1, p2, (100, c, 0))
        c += 50
    return image


def draw_boxes_on_image(data, image):
    for line in data:
        line = [int(num_as_str) for num_as_str in line[:-1]]
        image = draw_box_on_image(image, line)
    return image


def draw_boxes_on_image_and_show(image, boxes):
    for box in boxes:
        locations = list(zip(box[0::2], box[1::2]))
        for p1, p2 in zip(locations, locations[1:] + [locations[0]]):
            try:
                image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
                image = cv2.line(image, tuple(map(int, p1)), tuple(map(int, p2)), (100, 0, 0))
            except Exception as e:
                print(e)
    cv2.imshow('asd', image)
    cv2.waitKey()


def main():
    image = cv2.imread(IMAGE_PATH)
    with open(DATA_PATH, newline='') as csvfile:
        data = csv.reader(csvfile)
        image = draw_boxes_on_image(data, image)
    cv2.imshow('asd', image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
