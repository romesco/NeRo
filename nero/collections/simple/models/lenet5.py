# Copyright (c) 2020, romesco, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import pytorch_lightning as pl
import hydra

#from nero.core.classes.common import typecheck
from nero.core.neural_types import AxisKind, AxisType, ImageValue, LogprobsType, NeuralType
from nero.collections.example.modules.lenet5 import LeNet5Module
from nero.collections.example.datasets.mnist import MNISTDataset

from dataclasses import dataclass
from typing import Any, Optional, Dict
from omegaconf import MISSING

class MNISTLeNet5(pl.LightningModule):
    """
    The LeNet-5 convolutional model.
    """

    def __init__(self, cfg: MNISTLeNet5Conf = MNISTLeNet5Conf()):
        super().__init__()

        # Initialize datasets
        #TODO: investigate whether these calls can go back into a parent module
        self.setup_training_data(cfg)
        self.setup_validation_data(cfg)
        self.setup_test_data(cfg)

        # Initialize modules
        self.module = LeNet5Module(cfg.module)
        self.loss = hydra.utils.instantiate(cfg.loss, weight=torch.ones(10))
        self.optim = hydra.utils.instantiate(cfg.optim, params=self.parameters())

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns:
            :class:`LeNet5Module` input types.
        """
        return self.module.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns:
            :class:`LeNet5Module` output types.
        """
        return self.module.output_types

    #@typecheck()
    def forward(self, images):
        """ Propagates data by calling the module :class:`LeNet5Module` forward. """
        return self.module.forward(images=images)

    #TODO: investigate whether this hook can go back into a parent module
    def configure_optimizers(self):
        return self.optim 
            
    def setup_training_data(self, cfg):
        """ Creates dataset, wrap it with dataloader and return the latter """
        # Instantiate dataset.
        # needs configen class before instantiatable
        mnist_ds = MNISTDataset(cfg.dataset)

        # Configure data loader.
        train_dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=mnist_ds)
        self._train_dl = train_dataloader

    def setup_validation_data(self, cfg_data):
        """ Not implemented. """
        self._val_dl = None

    def setup_test_data(self, cfg_data):
        """ Not implemented. """
        self._test_dl = None

    def training_step(self, batch, batch_idx):
        """ Training step, calculate loss. """
        # "Unpack" the batch.
        _, images, targets, _ = batch

        predictions = self(images=images)

        loss = self.loss(predictions, targets)

        return {"loss": loss}
    
    def train_dataloader(self):
        """ Not implemented. """
        return self._train_dl

    def save_to(self, save_path: str):
        """ Not implemented. """
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """ Not implemented. """
        pass

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        """ Not implemented. """
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        """ Not implemented. """
        pass
