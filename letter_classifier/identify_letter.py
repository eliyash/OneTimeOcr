from __future__ import print_function

from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from letter_classifier.default_locations import DEFAULT_NETWORK_PATH
from letter_classifier.mnist_like_net import Net
from app.tools import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN


def image_loader(loader, image):
    image = loader(image).float()
    image = image.unsqueeze(0)
    var_image = Variable(torch.Tensor(image))
    return var_image


def identify_letters(image_path: str, locations, network_path=DEFAULT_NETWORK_PATH):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = Image.open(image_path)
    sm = torch.nn.Softmax()
    model = torch.load(network_path)  # type: Net
    model.to(device)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((60, 60)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    class_to_location_dict = defaultdict(lambda: set())
    for x, y in locations:
        letter_image = image.crop((x-BOX_WIDTH_MARGIN, y-BOX_HEIGHT_MARGIN, x+BOX_WIDTH_MARGIN, y+BOX_HEIGHT_MARGIN))
        var_image = image_loader(data_transforms, letter_image)
        var_image = var_image.to(device)
        output = model(var_image)
        probabilities = sm(output)
        probabilities = probabilities.cpu().detach().squeeze(0)
        letter_class = np.argmax(np.array(probabilities))
        class_to_location_dict[letter_class].add((x, y))

    letter_to_location_dict = {list(value)[0]: value for value in class_to_location_dict.values()}
    return letter_to_location_dict


def main():
    image = r"C:\Workspace\MyOCR\EAST\test_pages\test_gez good - Copy.jpg"
    identify_letters(image, {(100, 100)})


if __name__ == '__main__':
    main()
