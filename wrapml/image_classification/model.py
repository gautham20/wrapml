import torch
import timm
import numpy as np
import logging
import pytorch_lightning as pl
from collections import OrderedDict

from wrapml.utils import (
    if_none,
    freeze,
    unfreeze_and_add_param_group
)
from wrapml.optimizer import get_optimizer
from wrapml.scheduler import get_scheduler

from wrapml.image_classification.transforms import snapmix


class ImageClassificationModel(pl.LightningModule):
    def __init__(
        self, model_name, img_size, loss_function, learning_rate,
        optimizer_params=None, scheduler_params=None, **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.optimizer_params = if_none(optimizer_params, {})
        self.scheduler_params = if_none(scheduler_params, {})
        self.metrics = pl.metrics.Accuracy()
        self.pretrained = kwargs.get('pretrained', False)
        self.num_classes = kwargs.get('num_classes', 2)
        self.snapmix_pct = kwargs.get('snapmix_pct', 0)
        # epoch from which SnapMix can be applied
        self.snapmix_min_epoch = kwargs.get('snapmix_min_epoch', 0)
        # split model into backbone and classifier
        self.n_classifier_layers = kwargs.get('n_classifier_layers', 1)
        # milestones for freezing models, whenever milestones are passed it
        # is assumed that the feature_extractor is frozen beforehand
        self.train_milestones = kwargs.get('train_milestones', {})
        # keep track of unfreezed layers
        self.currently_unfreezed_n = None
        self.currently_unfreezed_module = None
        self.intialize_model()
        self.save_hyperparameters()

    def intialize_model(self):
        model = timm.create_model(
            self.model_name, 
            pretrained=self.pretrained, 
            num_classes=self.num_classes
        )
        _layers = list(model.children())
        self.feature_extractor = torch.nn.Sequential(*_layers[:-self.n_classifier_layers-1])
        self.pooling = _layers[-self.n_classifier_layers-1]
        self.classifier = torch.nn.Sequential(*_layers[-self.n_classifier_layers:])
        if len(self.train_milestones) > 0:
            freeze(self.feature_extractor, train_bn=True)
    

    # overwriting the pytorch Module train function
    def train(self, mode=True):
        super().train(mode=mode)
        if mode and len(self.train_milestones) > 0:
            if self.current_epoch < min(self.train_milestones.keys()):
                logging.info(f'epoch {self.current_epoch} - freeze feature_extracter')
                freeze(module=self.feature_extractor, train_bn=True)
            else:
                for i in sorted(self.train_milestones.keys(), reverse=True):
                    if self.current_epoch >= i:
                        milestone = self.train_milestones[i]
                        logging.info(f'train executing in epoch - {self.current_epoch}')
                        if milestone.get('freeze_until', None):
                            logging.info(f'epoch {self.current_epoch} - freeze until {milestone.get("freeze_until")}')
                            freeze(
                                module=self.feature_extractor,
                                freeze_until_module=milestone.get('freeze_until'),
                                train_bn=milestone.get('train_bn', True)
                            )
                        elif milestone.get('unfreeze_n_layers') != 'all':
                            freeze(
                                module=self.feature_extractor, 
                                n=milestone.get('unfreeze_n_layers', None),
                                train_bn=milestone.get('train_bn', True)
                            )
                        break

    # adding parameter group to optimizer
    def on_epoch_start(self):
        if len(self.train_milestones) > 0:
            optimizer = self.trainer.optimizers[0]
            logging.info(f'epoch - {self.current_epoch} on epoch start')
            for i in sorted(self.train_milestones.keys()):
                if self.current_epoch == i:
                    milestone = self.train_milestones[i]
                    unfreeze_n_layers = milestone.get('unfreeze_n_layers', 0)
                    unfreeze_n_layers = 0 if unfreeze_n_layers == 'all' else unfreeze_n_layers
                    freeze_until_module = milestone.get('freeze_until', None)
                    cf = if_none(self.currently_unfreezed_n, len(self.feature_extractor))
                    logging.info(f'epoch {i} - adding param group')
                    unfreeze_and_add_param_group(
                        module=self.feature_extractor[unfreeze_n_layers:cf],
                        unfreeze_start=freeze_until_module,
                        unfreeze_end=self.currently_unfreezed_module,
                        optimizer=optimizer,
                        train_bn=milestone.get('train_bn', True),
                        lr=milestone.get('lr', None)
                    )
                    self.currently_unfreezed_module = freeze_until_module
            n_grad_params = len(list(filter(lambda p: p.requires_grad, self.parameters())))
            logging.info(f'epoch - {self.current_epoch} n_grad_params - {n_grad_params}')

    def forward(self, X):
        return self.classifier(self.pooling(self.feature_extractor(X)))

    def training_step(self, batch, batch_idx):
        X, y = batch
        # snapmix currently supported only in efficientNet
        if (
            self.current_epoch >= self.snapmix_min_epoch
            and np.random.rand() <= self.snapmix_pct
            and 'efficientnet' in self.model_name
        ):
            X, y_a, y_b, lam_a, lam_b = snapmix(
                X, y, self.feature_extractor, self.classifier, self.img_size
            )
            y_hat = self.forward(X)
            loss_a = self.loss_function(y_hat, y_a)
            loss_b = self.loss_function(y_hat, y_b)
            loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
        else:
            y_hat = self.forward(X)
            loss = self.loss_function(y_hat, y)
        accuracy = self.metrics(y_hat.argmax(1), y)
        self.log('train_loss', loss)
        output = OrderedDict({'loss': loss,
                              'accuracy': accuracy,
                              })

        return output

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([output['loss']
                                       for output in outputs]).mean()
        train_acc_mean = torch.stack([output['accuracy']
                                      for output in outputs]).sum().float()
        train_acc_mean /= len(outputs)
        n_grad_params = len(list(filter(lambda p: p.requires_grad, self.parameters())))
        logging.info(f'epoch {self.current_epoch} train acc {train_acc_mean}')
        self.log_dict({'train_loss_epoch': train_loss_mean,
                        'train_acc': train_acc_mean,
                        'n_grad_params': n_grad_params,
                        'step': self.current_epoch})

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss_function(y_hat, y)
        accuracy = self.metrics(y_hat.argmax(1), y)
        return {'val_loss': loss,
                'accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()
        val_acc_mean = torch.stack([output['accuracy']
                                    for output in outputs]).sum().float()
        val_acc_mean /= len(outputs)
        logging.info(f'epoch {self.current_epoch} val acc {val_acc_mean}')
        self.log_dict({'val_loss': val_loss_mean,
                        'val_acc': val_acc_mean,
                        'step': self.current_epoch})

    def configure_optimizers(self):
        if len(self.optimizer_params) == 0:
            self.optimizer_params = {
                'optimizer': 'AdamW',
                'lr': self.learning_rate
            }
        self.optimizer_params['lr'] = self.learning_rate
        optimizer = get_optimizer(self.optimizer_params)(
            filter(lambda p: p.requires_grad, self.parameters())
        )
        scheduler = get_scheduler(self.scheduler_params)(optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': self.scheduler_params.get('interval', 'epoch')
        }
        if 'monitor' in self.scheduler_params:
            scheduler_dict.update({'monitor': self.scheduler_params['monitor']})
        return [optimizer], [scheduler_dict]