import torch
import pytorch_lightning as pl


from wrapml.utils import if_none
from wrapml.optimizer import get_optimizer
from wrapml.scheduler import get_scheduler

class ImageClassificationModule(pl.LightningModule):
    def __init__(
        self, model, loss_function, learning_rate,
        optimizer_params=None, scheduler_params=None
    ):
        super(ImageClassificationModule, self).__init__()
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.optimizer_params = optimizer_params
        self.scheduler_params = if_none(scheduler_params, {})
        self.metrics = pl.metrics.Accuracy()

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss_function(y_hat, y)
        accuracy = self.metrics(y_hat.argmax(1), y)
        self.log(
            'train_loss', loss, on_step=True, logger=True
        )
        self.log(
            'train_accuracy', accuracy, on_step=True,
            on_epoch=True, logger=True, prog_bar=False
        )
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss_function(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True,logger=True)
        accuracy = self.metrics(y_hat.argmax(1), y)
        self.log(
            'val_accuracy', accuracy, on_step=False, 
            on_epoch=True, logger=True
        )
        return {'val_loss': loss}

    # def validation_epoch_end(self, outputs):
    #     avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     self.log(
    #         'val_loss', avg_val_loss, on_epoch=True, 
    #         prog_bar=True, logger=True
    #     )
    #     return {'val_loss': avg_val_loss}

    def configure_optimizers(self):
        if self.optimizer_params is None:
            self.optimizer_params = {
                'optimizer': 'AdamW',
                'lr': self.learning_rate
            }
        optimizer = get_optimizer(self.optimizer_params)(self.model.parameters())
        scheduler = get_scheduler(self.scheduler_params)(optimizer)
        return [optimizer], [{
            'scheduler': scheduler,
            'interval': self.scheduler_params.get('interval', 'epoch'),
            'monitor': 'val_loss'
        }]
    