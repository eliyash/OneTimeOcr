import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
import time
import matplotlib.pyplot as plt
import numpy as np

from app.paths import TRAIN_DATA_PATH, IMAGES_PATH, DETECTOR_NETS_PATH, LETTERS_PATH
from app.tools import get_device, plot_train_losses
from letter_detector.dataset import CustomDataset
from letter_detector.detect import get_boxes
from letter_detector.draw_gt import draw_boxes_on_image_and_show
from letter_detector.loss import Loss
from letter_detector.model import EAST

BASIC_EAST_MODEL = '../networks/pretrained/east_vgg16.pth'
DEFAULT_NUMBER_OF_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-3

def show_detection_image(img, pred_geo, pred_score):
    for single_pred_score, single_pred_geo, single_img in zip(
            pred_score.detach().cpu().numpy(),
            pred_geo.detach().cpu().numpy(),
            np.moveaxis(img.cpu().numpy().copy(), 1, -1)
    ):
        boxes = get_boxes(single_pred_score, single_pred_geo)
        if boxes is not None:
            draw_boxes_on_image_and_show(single_img, boxes)


def train(
        images_path: Path, gt_data_path: Path, train_portion: float, networks_path: Path, batch_size, lr, num_workers,
        number_of_epochs, minimal_improvement_for_save, start_net=None, set_new_train_fig: Callable = None,
        show_detection=False):
    gt_path = gt_data_path / 'pages'
    device = get_device()
    data_set = CustomDataset(images_path, gt_path, duplicate_pages=batch_size)

    size = len(data_set)
    if size < 2 * batch_size:
        train_set, test_set = data_set, data_set
    else:
        train_size = int(size * train_portion)
        train_set, test_set = torch.utils.data.random_split(data_set, [train_size, size-train_size])

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader_dynamic = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    iterator = iter(test_loader_dynamic)
    test_loader = [next(iterator) for _ in range(len(test_loader_dynamic))]

    file_num = len(train_set)

    criterion = Loss()

    model = EAST()
    if start_net:
        model.load_state_dict(torch.load(start_net))  # added

    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[number_of_epochs // 2], gamma=0.1)

    train_losses_by_epochs = []
    test_losses_by_epochs = []

    best_training_loss = None
    for epoch in range(number_of_epochs):
        model.train()
        epoch_train_losses = []
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            loss, pred_score, pred_geo = calc_loss_on_net(criterion, device, gt_geo, gt_score, ignored_map, img, model)
            epoch_train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
                epoch + 1, number_of_epochs, i + 1, int(file_num / batch_size), time.time() - start_time, loss.item())
            )
            if show_detection:
                print('pred')
                show_detection_image(img, pred_geo, pred_score)
                print('gt')
                show_detection_image(img, gt_geo, gt_score)
        scheduler.step(epoch)

        model.eval()
        epoch_test_losses = []
        with torch.no_grad():
            for i, (img, gt_score, gt_geo, ignored_map) in enumerate(test_loader):
                loss, pred_score, pred_geo = calc_loss_on_net(criterion, device, gt_geo, gt_score, ignored_map, img, model)
                epoch_test_losses.append(loss.item())

        train_avg_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        test_avg_loss = sum(epoch_test_losses) / len(epoch_test_losses)

        train_losses_by_epochs.append(train_avg_loss)
        test_losses_by_epochs.append(test_avg_loss)

        print('test_loss is {:.8f}, train_loss is {:.8f}, epoch_time is {:.8f}'.format(
            test_avg_loss, train_avg_loss, time.time() - epoch_time)
        )
        print(time.asctime(time.localtime(time.time())))
        if best_training_loss is None or test_avg_loss < (best_training_loss - minimal_improvement_for_save):
            best_training_loss = test_avg_loss
            new_net_name = 'model_epoch_{}.pth'.format(epoch + 1)
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, str(networks_path / new_net_name))
            print('new net saved, name: {}, loss: {}'.format(new_net_name, test_avg_loss))
            with open(str(networks_path / 'state.json'), 'w') as state_file:
                json.dump({'loss': best_training_loss}, state_file)
        print('=' * 80)

        if set_new_train_fig:
            set_new_train_fig(test_losses_by_epochs, train_losses_by_epochs, 'Letter Detector Loss')


def calc_loss_on_net(criterion, device, gt_geo, gt_score, ignored_map, img, model):
    img, gt_score, gt_geo, ignored_map = \
        img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
    pred_score, pred_geo = model(img)
    loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
    return loss, pred_score, pred_geo


def run_train(
        data_set_path,
        number_of_epochs=DEFAULT_NUMBER_OF_EPOCHS,
        lr=DEFAULT_LEARNING_RATE,
        set_new_train_fig: Callable = None):
    torch.manual_seed(1)
    train_portion = 0.5
    start_net = BASIC_EAST_MODEL
    batch_size = 4
    num_workers = 4
    minimal_improve_to_save = 0
    network_path = DETECTOR_NETS_PATH / time.strftime("train_%Y%m%d-%H%M%S")
    Path(network_path).mkdir(parents=True)
    train(
        IMAGES_PATH, data_set_path, train_portion, network_path, batch_size, lr,
        num_workers, number_of_epochs, minimal_improve_to_save, start_net, set_new_train_fig, show_detection=True
    )


def main():
    fig = plt.figure()
    # You probably won't need this if you're embedding things in a tkinter plot...
    plt.ion()
    subplot = fig.add_subplot()

    def set_new_train_fig(test_losses_by_epochs, train_losses_by_epochs, title):
        plt.clf()
        x = range(len(train_losses_by_epochs))
        subplot.clear()
        subplot.set_xlabel('epoch', fontsize=12)
        subplot.set_ylabel('loss', fontsize=12)
        subplot.set_title(title)
        subplot.plot(x, train_losses_by_epochs, 'b', x, test_losses_by_epochs, 'r')
        plt.show()

    data_set_path = Path(r'C:\Workspace\MyOCR\data\annotations\dataset_20211126-095235')
    run_train(data_set_path, set_new_train_fig=set_new_train_fig)


if __name__ == '__main__':
    main()
