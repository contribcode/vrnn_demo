from typing import NamedTuple

import torch


class Config:

    def __init__(self) -> None:
        self.general = GeneralConf(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            seed=42
        )
        self.data = DataConf(x_dim=1, num_classes=10, valid_size=10_000)
        self.vrnn = VrnnConf(x_ft=16, h_dim=512, z_dim=64, z_ft=128)
        self.train = TrainConf(
            batch_size=64,
            lr_vrnn=3e-5,
            lr=1e-3,
            max_epochs=10,
            val_check_interval=0.1,
            patience=10,
            log_every_n_steps=10
        )


class GeneralConf(NamedTuple):
    device: str
    seed: int

class DataConf(NamedTuple):
    x_dim: int
    num_classes: int
    valid_size: float | int

class VrnnConf(NamedTuple):
    x_ft: int
    h_dim: int
    z_dim: int
    z_ft: int

class TrainConf(NamedTuple):
    batch_size: int
    lr_vrnn: float
    lr: float
    max_epochs: int
    val_check_interval: float
    patience: int
    log_every_n_steps: int
