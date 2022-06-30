import csv
import cv2
import numpy as np


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


def draw_boxes_on_image_and_show(single_img, single_pred_score):
    # single_img[::4, ::4], single_pred_score
    # for box in boxes:
    #     locations = list(zip(box[0::2], box[1::2]))
    #     for p1, p2 in zip(locations, locations[1:] + [locations[0]]):
    #         try:
    #             image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
    #             image = cv2.line(image, tuple(map(int, p1)), tuple(map(int, p2)), (100, 0, 0))
    #         except Exception as e:
    #             print(e)
    single_img = cv2.cvtColor(cv2.cvtColor(np.array(single_img), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
    mask_gray = cv2.resize(single_pred_score.astype(np.uint8), tuple(np.array(list(reversed(single_pred_score.shape)))*4))
    cv2.imwrite(r'c:\temp\ocr_blob.jpg', mask_gray)

    mask = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB) * 100
    mask[:, :, 1] = 0
    im = single_img - mask[:single_img.shape[0], :single_img.shape[1]]
    im[im < 0] = 0
    # image = single_img[::4, ::4] + single_pred_score[0]
    cv2.imshow('im', single_img)
    cv2.imshow('with_mask', im)
    cv2.waitKey()


def main():
    image = cv2.imread(IMAGE_PATH)
    with open(DATA_PATH, newline='') as csvfile:
        data = csv.reader(csvfile)
        image = draw_boxes_on_image(data, image)
    cv2.imshow('asd', image)
    cv2.waitKey()


def test_blob():
    img = cv2.imread(r'c:\temp\ocr_blob.jpg', cv2.IMREAD_GRAYSCALE)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(img)

    label_hue = np.uint8(179 * labels_im / np.max(labels_im))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    points_image = np.zeros_like(img)
    r = 3
    for x, y in centroids.astype(np.int):
        points_image[y-r:y+r, x-r:x+r] = 100
    print(len(centroids))

    cv2.imshow('points img', points_image)
    cv2.imshow('img', img * 100)
    cv2.imshow('labeled img', labeled_img)
    cv2.waitKey()


if __name__ == '__main__':
    test_blob()
    # main()
