import os
import math
import glob
import torch
import numpy as np
from PIL import Image
from torch import optim
import torchvision
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.transforms import Pad
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.save_hyperparameters(params)
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except BaseException:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(
            *
            results,
            M_N=self.params['batch_size'] /
            self.num_train_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item()
                                    for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
            *
            results,
            M_N=self.params['batch_size'] /
            self.num_val_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx)

        real_img = Pad(padding=(2, 1))(real_img)
        results[0] = Pad(padding=(2, 1))(results[0])

        n = min(len(batch[0]), 8)
        comparison = torch.cat([real_img[:n], results[0][:n]])
        self.logger.experiment.add_images(
            'generated_images_epoch=' + str(self.current_epoch + 1), comparison, batch_idx)

        # self.freeze()
        # x, img = batch
        # x = self.forward(x)
        # n = min(len(batch[0]), 8)
        # comparison = torch.cat([
        #     img.view(len(x), 1, self.data_shape_size, self.data_shape_size)[:n],
        # x.view(len(x), 1, self.data_shape_size, self.data_shape_size)[:n]])

        # self.logger.experiment.add_images(
        #     'generated_images_epoch=' + str(self.current_epoch + 1), comparison, batch_idx)
        # self.unfreeze()

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(
            recons.data,
            f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
            f"recons_{self.logger.name}_{self.current_epoch}.png",
            normalize=True,
            nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(
                samples.cpu().data,
                f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                f"{self.logger.name}_{self.current_epoch}.png",
                normalize=True,
                nrow=12)
        except BaseException:
            pass

        del test_input, recons  # , samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial
        # training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(
                    getattr(
                        self.model,
                        self.params['submodel']).parameters(),
                    lr=self.params['LR_2'])
                optims.append(optimizer2)
        except BaseException:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optims[0], gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second
                # optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(
                            optims[1], gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except BaseException:
                    pass
                return optims, scheds
        except BaseException:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root=self.params['data_path'],
                             split="train",
                             transform=transform,
                             download=False)
        elif self.params['dataset'] == 'speckle':
            # TODO prepare dataset or import dataset class
            dataset = S2SDataSet(params=self.params)
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            self.sample_dataloader = DataLoader(
                CelebA(
                    root=self.params['data_path'],
                    split="test",
                    transform=transform,
                    download=True),
                batch_size=144,
                shuffle=True,
                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'speckle':
            # TODO prepare dataset or import dataset class
            dataset = S2SDataSet(params=self.params)
            self.num_val_imgs = len(dataset)
            return DataLoader(
                dataset,
                batch_size=self.params['batch_size'],
                shuffle=False,
                drop_last=True)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            return None
        return transform


class S2SDataSet(torch.utils.data.Dataset):
    """
        Surface to speckle image deep learning.
        data_dir_path :list(str): path of the data directories. you can use data in multiple directories.
    """

    def __init__(self, params):
        # label data
        self.shape_size = int(params["img_size"])
        self.jpg_path_list = []
        self.jpg_path_list += glob.glob(
            os.path.join(params["data_path"], "*.jpg"))
        self.jpg_path_list = self.jpg_path_list[:int(params["data_num"])]

        print("\nlength of dataset is {}\n\n".format(len(self.jpg_path_list)))

        self.transform = torchvision.transforms.Compose([
            transforms.CenterCrop(1536),
            transforms.Resize(self.shape_size),
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ])

    def __len__(self):
        return len(self.jpg_path_list)

    def __getitem__(self, idx):
        data = Image.open(self.jpg_path_list[idx])
        return self.transform(data).float(), 1
