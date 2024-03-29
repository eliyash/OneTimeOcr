from __future__ import print_function

import json
import shutil
import time
from typing import Callable

import cv2
import torch
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from app.paths import TRAIN_DATA_PATH, IDENTIFIER_NETS_PATH
from app.tools import get_device
from letter_classifier.mnist_like_net import Net

DEFAULT_NUMBER_OF_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.001


def train(model, device, train_loader, optimizer, epoch):
    sum_train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        sum_train_loss += loss.item()
        optimizer.step()
        print('Train Epoch: {}, Batch: {}'.format(epoch, batch_idx))
    return sum_train_loss / len(train_loader.dataset)


def test(model, device, test_loader, epoch, output_folder):
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            data, pred = data.cpu(), pred.cpu()

            images = [pytorch_image.numpy()[0] for pytorch_image in data]
            inferences = [pytorch_inference.numpy() for pytorch_inference in pred]
            for image, inference in zip(images, inferences):
                path = output_folder / 'test_images' / str(epoch) / str(inference)
                path.mkdir(exist_ok=True, parents=True)
                image_as_uint = (image*255).astype('uint8')
                cv2.imwrite(str(path / r'{}.jpg'.format(counter)), image_as_uint)
                counter += 1

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def run_train(
        data_set_path,
        number_of_epochs=DEFAULT_NUMBER_OF_EPOCHS,
        lr=DEFAULT_LEARNING_RATE,
        set_new_train_fig: Callable = None
):
    torch.manual_seed(1)
    device = get_device()

    train_portion = 0.7
    batch = 8
    network_path = IDENTIFIER_NETS_PATH / time.strftime("train_%Y%m%d-%H%M%S")
    network_path.mkdir(parents=True)

    transform = transforms.Compose([
        transforms.Resize((60, 60)),
        transforms.RandomRotation(10),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    image_net_data = ImageFolder(data_set_path / 'letter_images', transform=transform)

    number_of_classes = len(image_net_data.classes)
    with open(str(network_path / 'classes_map.json'), 'w') as state_file:
        json.dump({ind: class_name for ind, class_name in enumerate(image_net_data.classes)}, state_file)

    shutil.copytree(str(data_set_path / 'letters_map'), str(network_path / 'letters_map'))
    size = len(image_net_data)
    train_size = int(size * train_portion)
    train_set_data, test_set_data = torch.utils.data.random_split(image_net_data, [train_size, size-train_size])

    train_loader = torch.utils.data.DataLoader(train_set_data, batch_size=batch, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_set_data, batch_size=batch, shuffle=True, num_workers=1)

    model = Net(number_of_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

    best_test_loss = None
    test_losses_by_epochs = []
    train_losses_by_epochs = []
    for epoch in range(number_of_epochs):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader, epoch, network_path)
        train_losses_by_epochs.append(train_loss)
        test_losses_by_epochs.append(test_loss)
        if best_test_loss is None or test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model, str(network_path / 'epoch_{}.pth'.format(epoch)))
            with open(str(network_path / 'state.json'), 'w') as state_file:
                json.dump({'loss': best_test_loss}, state_file)

        if set_new_train_fig:
            set_new_train_fig(test_losses_by_epochs, train_losses_by_epochs, 'Letter Identifier Loss')

    torch.save(model, str(network_path / 'last_net.pth'))


def main():
    data_set_path = TRAIN_DATA_PATH / 'dataset_20201209-145735'
    run_train(data_set_path)


if __name__ == '__main__':
    main()
