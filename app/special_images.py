import cv2
import numpy as np

from app.special_values import BOX_HEIGHT_MARGIN, BOX_WIDTH_MARGIN

EMPTY_IMAGE = np.ones((BOX_HEIGHT_MARGIN*2, BOX_WIDTH_MARGIN*2)) * 255
UNKNOWN_IMAGE = cv2.line(
    cv2.circle(EMPTY_IMAGE.copy(), (BOX_HEIGHT_MARGIN, BOX_WIDTH_MARGIN), 15, 0, 2),
    (0, 0), (BOX_HEIGHT_MARGIN*2, BOX_WIDTH_MARGIN*2), 0, 2
)
