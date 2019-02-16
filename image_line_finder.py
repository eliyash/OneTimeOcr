import cv2
import numpy as np


def get_connected_components(src):
    # Threshold it so it becomes binary
    ret, thresh = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thresh = cv2.Canny(thresh, 100, 200, apertureSize=3)
    # cv2.imshow('edges', thresh)
    # cv2.waitKey(0)

    cv2.imshow('thresh', thresh)

    # You need to choose 4 or 8 for connectivity type
    connectivity = 4
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    centroids_image = np.zeros_like(src)

    for centroid in centroids:
        cv2.circle(centroids_image, (int(centroid[0]), int(centroid[1])), 5, (255, 255, 255), -1)
        # centroids_image[int(centroid[1]), int(centroid[0])] = 180
    cv2.imshow('centroids', centroids_image)
    return centroids_image


def export_potential_lines(image):
    # look for connected elements and make them as a dot

    # edges = cv2.Canny(image, 100, 200, apertureSize=3)
    edges = image
    cv2.imshow('edges', edges)

    minLineLength = 20
    maxLineGap = 0
    # HoughLinesPointSet
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    # if lines:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 100), 2)

    cv2.imshow('hough', img)
    cv2.waitKey(0)

    # check with half transform where the lines are



img = cv2.imread(r"C:\Users\eli\Dropbox\Workspace\AI\mnist\text1.png")
cv2.imshow('img', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray', gray)

centroids = get_connected_components(gray)

export_potential_lines(centroids)

cv2.waitKey(0)

while True:
    pass