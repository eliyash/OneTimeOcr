import json
from pathlib import Path

import torch
from torch.utils import data, tensorboard
from torch.optim import lr_scheduler
import time
import matplotlib.pyplot as plt
import numpy as np

from app.paths import IMAGES_PATH, DETECTOR_NETS_PATH
from app.tools import get_device
from letter_detector.dataset import BaseDataset, TilesDataset, RandomTilesDataset
from letter_detector.draw_gt import draw_boxes_on_image_and_show
from letter_detector.loss import Loss
from letter_detector.model import EAST

BASIC_EAST_MODEL = '../networks/pretrained/east_vgg16.pth'
DEFAULT_NUMBER_OF_EPOCHS = 200
DEFAULT_LEARNING_RATE = 1e-3


def show_detection_image(img, mask):
    for score_mask, single_img in zip(
            mask.detach().cpu().numpy(),
            np.moveaxis(img.cpu().numpy().copy(), 1, -1)
    ):

        draw_boxes_on_image_and_show(single_img, score_mask[0])


def train(
        images_path: Path, gt_data_path: Path, train_portion: float, networks_path: Path, batch_size, lr, num_workers,
        number_of_epochs, minimal_improvement_for_save, start_net=None, show_detection=False):
    gt_path = gt_data_path / 'pages'
    device = get_device()
    train_set = RandomTilesDataset(images_path, gt_path)
    test_set = TilesDataset(images_path, gt_path)

    summery = tensorboard.SummaryWriter(str(networks_path))

    number_of_iter = 20

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
    )

    file_num = len(train_set)
    num_of_batches = int(file_num / batch_size)

    criterion = Loss()

    model = EAST()
    # if start_net:
    #     model.load_state_dict(torch.load(start_net))  # added

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[number_of_epochs // 4], gamma=0.6)

    train_losses_by_epochs = []
    test_losses_by_epochs = []

    best_training_loss = None
    for epoch in range(number_of_epochs):
        model.train()
        epoch_train_losses = []
        epoch_time = time.time()
        for i in range(number_of_iter):
            for img, gt_score in train_loader:
                start_time = time.time()
                loss, pred_score = calc_loss_on_net(criterion, device, gt_score, img, model)
                epoch_train_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
                    epoch + 1, number_of_epochs, i + 1, number_of_iter, time.time() - start_time, loss.item())
                )
                # if show_detection:
                #     print('gt')
                #     show_detection_image(img, gt_score)
                #     print('pred')
                #     show_detection_image(img, pred_score)
        scheduler.step(epoch)

        model.eval()
        epoch_test_losses = []
        with torch.no_grad():
            for i, (img, gt_score) in enumerate(test_loader):
                loss, pred_score = calc_loss_on_net(criterion, device, gt_score, img, model)
                epoch_test_losses.append(loss.item())

        train_avg_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        test_avg_loss = sum(epoch_test_losses) / len(epoch_test_losses)

        summery.add_scalar(tag='train', scalar_value=train_avg_loss, global_step=epoch)
        summery.add_scalar(tag='validation', scalar_value=test_avg_loss, global_step=epoch)

        train_losses_by_epochs.append(train_avg_loss)
        test_losses_by_epochs.append(test_avg_loss)

        print('test_loss is {:.8f}, train_loss is {:.8f}, epoch_time is {:.8f}'.format(
            test_avg_loss, train_avg_loss, time.time() - epoch_time)
        )
        print(time.asctime(time.localtime(time.time())))
        if best_training_loss is None or test_avg_loss < (best_training_loss - minimal_improvement_for_save):
            best_training_loss = test_avg_loss
            new_net_name = 'model_epoch_{}.pth'.format(epoch + 1)
            torch.save(model.state_dict(), str(networks_path / new_net_name))

            print('new net saved, name: {}, loss: {}'.format(new_net_name, test_avg_loss))
            state_json = networks_path / 'state.json'
            state_json.write_text(json.dumps({'loss': best_training_loss}))
        print('=' * 80)


def calc_loss_on_net(criterion, device, gt_score, img, model):
    img, gt_score = list(map(lambda x: x.to(device), [img, gt_score]))
    pred_score = model(img)
    loss = criterion(gt_score, pred_score)
    return loss, pred_score


def run_train(data_set_path, number_of_epochs=DEFAULT_NUMBER_OF_EPOCHS, lr=DEFAULT_LEARNING_RATE):
    torch.manual_seed(1)
    train_portion = 0.5
    start_net = BASIC_EAST_MODEL
    batch_size = 64
    num_workers = 4
    minimal_improve_to_save = 0
    network_path = DETECTOR_NETS_PATH / time.strftime("train_%Y%m%d-%H%M%S")
    Path(network_path).mkdir(parents=True)
    train(
        IMAGES_PATH, data_set_path, train_portion, network_path, batch_size, lr,
        num_workers, number_of_epochs, minimal_improve_to_save, start_net, show_detection=True
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

    data_set_path = Path(r'C:\Workspace\MyOCR\data\annotations\dataset_20211202-173416')
    run_train(data_set_path, 50)  # , lr=0.01)


if __name__ == '__main__':
    main()
