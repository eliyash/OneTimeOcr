from __future__ import print_function

import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from letter_classifier.default_locations import DEFAULT_IMAGES_FOLDER, DEFAULT_NETWORK_PATH
from letter_classifier.mnist_like_net import Net


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {}, Batch: {}'.format(epoch, batch_idx))


def test(model, device, test_loader, epoch, benchmark_folder):
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
                path = benchmark_folder / str(epoch) / str(inference)
                path.mkdir(exist_ok=True, parents=True)
                image_as_uint = (image*255).astype('uint8')
                cv2.imwrite(str(path / r'{}.jpg'.format(counter)), image_as_uint)
                counter += 1

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def run_train(images_folder, output_path):
    benchmark_folder = Path(r'C:\temp\train') / time.strftime("%Y%m%d-%H%M%S")
    number_of_epochs = 100
    torch.manual_seed(1)

    device = "cuda"

    train_transform = transforms.Compose([
        transforms.Resize((60, 60)),
        transforms.RandomRotation(10),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((60, 60)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    image_net_train_data = ImageFolder(images_folder, transform=train_transform)
    image_net_test_data = ImageFolder(images_folder, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(image_net_train_data,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=1)

    test_loader = torch.utils.data.DataLoader(image_net_test_data,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=1)

    model = Net(len(image_net_train_data.classes)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(number_of_epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch, benchmark_folder)

    torch.save(model, output_path)


def main():
    run_train(DEFAULT_IMAGES_FOLDER, DEFAULT_NETWORK_PATH)


if __name__ == '__main__':
    main()

