from pathlib import Path

from torch.utils.data import random_split

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from config import Config
from lightning_module import LitClf


conf = Config()
# data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()),
        transforms.Lambda(lambda x: x.unsqueeze(dim=1))
    ]
)
train_dataset = datasets.MNIST(
    root='data', train=True, download=True, transform=transform
)
valid_size = conf.data.valid_size
train_size = len(train_dataset) if isinstance(valid_size, int) else 1
train_dataset, valid_dataset = random_split(
    train_dataset, lengths=[train_size - valid_size, valid_size]
)
test_dataset = datasets.MNIST(
    root='data', train=False, download=True, transform=transform
)
train_ldr = DataLoader(
    dataset=train_dataset, batch_size=conf.train.batch_size, shuffle=True
)
valid_ldr = DataLoader(dataset=valid_dataset, batch_size=conf.train.batch_size)
test_ldr = DataLoader(dataset=test_dataset, batch_size=conf.train.batch_size)
# model
clf = LitClf(conf=conf)
# train
# callbacks
early_stopping_cb = EarlyStopping(
    monitor='valid_f1', patience=conf.train.patience, mode='max'
)
ckpt_cb = ModelCheckpoint(monitor='valid_f1', mode='max')
# loggers
csv_logger = CSVLogger(Path.cwd())
wandb_logger = WandbLogger(project='demo-vrnn', log_model=False)
trainer = L.Trainer(
    logger=[csv_logger, wandb_logger],
    callbacks=[early_stopping_cb, ckpt_cb],
    val_check_interval=conf.train.val_check_interval,
    max_epochs=conf.train.max_epochs,
    log_every_n_steps=conf.train.log_every_n_steps
)
trainer.fit(model=clf, train_dataloaders=train_ldr, val_dataloaders=valid_ldr)
trainer.test(dataloaders=test_ldr, ckpt_path='best')
