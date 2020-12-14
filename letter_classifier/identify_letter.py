from __future__ import print_function
import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms
from letter_classifier.mnist_like_net import Net
from app.tools import str_to_location, file_name_to_location, get_last_epoch_net, get_device, box_lines_from_center


def image_loader(loader, image):
    image = loader(image).float()
    image = image.unsqueeze(0)
    var_image = Variable(torch.Tensor(image))
    return var_image


def identify_letters(image_path: str, locations, network_path: Path, image_shape: Tuple):
    model_path = get_last_epoch_net(network_path)

    device = get_device()

    image = Image.open(image_path)
    softmax = torch.nn.Softmax(dim=1)
    model = torch.load(model_path)  # type: Net
    model.to(device)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((60, 60)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    class_to_location_dict = defaultdict(lambda: set())
    for x, y in locations:
        letter_image = image.crop(tuple(box_lines_from_center((x, y), image_shape)))
        var_image = image_loader(data_transforms, letter_image)
        var_image = var_image.to(device)
        output = model(var_image)
        probabilities = softmax(output)
        probabilities = probabilities.cpu().detach().squeeze(0)
        letter_class = np.argmax(np.array(probabilities))
        class_to_location_dict[letter_class].add((x, y))

    with open(str(network_path / 'classes_map.json'), 'r') as state_file:
        classes_map = json.load(state_file)

    letter_to_location_dict = {str_to_location(classes_map[str(key)]): value
                               for key, value in class_to_location_dict.items()}

    letters_map = {file_name_to_location(letter_file): cv2.imread(str(letter_file))
                   for letter_file in (network_path / 'letters_map').iterdir()}

    return dict(letter_to_location_dict), letters_map


def main():
    image = r"C:\Workspace\MyOCR\data\images\test_gez good - Copy.jpg"
    train_path = Path(r'C:\Workspace\MyOCR\identifieng letters\networks\identifier\train_20201209-160931')
    identify_letters(image, {(100, 100)}, train_path, (50, 50))


if __name__ == '__main__':
    main()
