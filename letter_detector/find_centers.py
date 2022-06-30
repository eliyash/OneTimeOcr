from pathlib import Path

import torch
import numpy as np
from PIL import Image
from app.tools import get_device, get_last_epoch_net
from letter_detector.detect import detect
from letter_detector.model import EAST


def detect_letters(img_path, network_path):
    model_path = get_last_epoch_net(network_path)
    device = get_device()
    model = EAST().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    img = Image.open(img_path)
    
    locations = detect(img, model, device)

    # locations = set()
    # for box in list(boxes):
    #     locations.add((int(np.mean(box[0:-1:2])), int(np.mean(box[1::2]))))

    return locations


def main():
    image_path = r"..\..\test_pages\test_gez good - Copy.jpg"
    train_path = Path("../networks/letter_detector")
    detect_letters(image_path, train_path)


if __name__ == '__main__':
    main()
