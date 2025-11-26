from typing import Any
import torch
import torch.nn as nn
from torch.optim import AdamW

import lightning as L
from torchmetrics.classification import MulticlassF1Score

from config import Config
from model import VRNN


class LitClf(L.LightningModule):
    
    def __init__(self, conf : Config) -> None:
        super(LitClf, self).__init__()
        self.conf = conf
        self.rnn_model = VRNN(conf=conf)
        self.clf = nn.Linear(
            in_features=conf.vrnn.z_dim, out_features=conf.data.num_classes
        )
        # loss
        self.clf_criterion = nn.CrossEntropyLoss()
        # metric
        self.train_f1 = MulticlassF1Score(num_classes=conf.data.num_classes)
        self.valid_f1 = MulticlassF1Score(num_classes=conf.data.num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=conf.data.num_classes)

    def forward(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_seq, kld_loss, nll_loss = self.rnn_model(data)
        clf_out = self.clf(latent_seq[:,-1,:])
        return kld_loss, nll_loss, clf_out

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data, target = batch
        kld_loss, nll_loss, clf_out = self(data=data)
        # losses
        clf_loss = self.clf_criterion(clf_out, target)
        loss = torch.stack(tensors=[kld_loss, nll_loss, clf_loss]).mean(dim=0)
        self.log(name='train_kld', value=kld_loss, batch_size=data.shape[0])
        self.log(name='train_nll', value=nll_loss, batch_size=data.shape[0])
        self.log(name='train_clf', value=clf_loss, batch_size=data.shape[0])
        # metric
        self.train_f1(preds=clf_out, target=target)
        self.log(
            name='train_f1',
            value=self.train_f1,
            prog_bar=True,
            on_epoch=True,
            batch_size=data.shape[0]
        )
        return loss

    def configure_optimizers(self) -> AdamW:
        params_config = [
            {'params': self.rnn_model.parameters(), 'lr': self.conf.train.lr_vrnn},
            {'params': self.clf.parameters()}
        ]
        optimizer = AdamW(params=params_config, lr=self.conf.train.lr)
        return optimizer

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data, target = batch
        kld_loss, nll_loss, clf_out = self(data=data)
        # losses
        clf_loss = self.clf_criterion(clf_out, target)
        loss = torch.stack(tensors=[kld_loss, nll_loss, clf_loss]).mean(dim=0)
        self.log(name='valid_kld', value=kld_loss, batch_size=data.shape[0])
        self.log(name='valid_nll', value=nll_loss, batch_size=data.shape[0])
        self.log(name='valid_clf', value=clf_loss, batch_size=data.shape[0])
        # metric
        self.valid_f1(preds=clf_out, target=target)
        self.log(
            name='valid_f1',
            value=self.valid_f1,
            prog_bar=True,
            on_epoch=True,
            batch_size=data.shape[0]
        )
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, target = batch
        kld_loss, nll_loss, clf_out = self(data=data)
        # losses
        clf_loss = self.clf_criterion(clf_out, target)
        loss = torch.stack(tensors=[kld_loss, nll_loss, clf_loss]).mean(dim=0)
        self.log(
            name='test_kld',
            value=kld_loss,
            on_step=False,
            on_epoch=True,
            batch_size=data.shape[0]
        )
        self.log(
            name='test_nll',
            value=nll_loss,
            on_step=False,
            on_epoch=True,
            batch_size=data.shape[0]
        )
        self.log(
            name='test_clf',
            value=clf_loss,
            on_step=False,
            on_epoch=True,
            batch_size=data.shape[0]
        )
        # metric
        self.test_f1(preds=clf_out, target=target)
        self.log(
            name='test_f1',
            value=self.test_f1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=data.shape[0]
        )
        return loss
