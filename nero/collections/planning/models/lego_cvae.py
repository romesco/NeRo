import omegaconf
from omegaconf import MISSING
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl
from nemo.core import ModelPT
from nemo.core.classes.common import typecheck
from nero.collections.common.mlp import MultiLayerPerceptron
from nemo.core.neural_types import AxisKind, AxisType, ImageValue, LogprobsType, NeuralType
from nemo.core.neural_types import NormalDistributionMeanType

import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from hydra_configs.pytorch_lightning.trainer import TrainerConf

#TODO: can we make this into VAE
class LEGO(ModelPT):
    #TODO: change this to be actual signature 
    def __init__(self, cfg) :
        super().__init__(cfg=cfg)

        # TODO: hydra_instantiate? or from_config_dict?
        self.encoder = self.from_config_dict(self.cfg.encoder)
        self.fc_mu = nn.Linear(self.cfg.encoder.num_classes, self.cfg.latent_dim)
        self.fc_var = nn.Linear(self.cfg.encoder.num_classes, self.cfg.latent_dim)
        self.decoder = self.from_config_dict(self.cfg.decoder)

    
    @property
    def input_types(self):
        #return self.encoder.input_types 
        return {
            "z": NeuralType(
                axes=(
                    AxisType(kind=AxisKind.Batch),
                    AxisType(kind=AxisKind.Any),
                ),
                elements_type=NormalDistributionMeanType(),
            )
        }


    @property
    def output_types(self):
        #return self.decoder.output_types 
        return {
            "images": NeuralType(
                axes=(
                    AxisType(kind=AxisKind.Batch),
                    AxisType(kind=AxisKind.Channel, size=100),
                    #AxisType(kind=AxisKind.Height, size=100),
                    #AxisType(kind=AxisKind.Width, size=100),
                ),
                elements_type=ImageValue(),
            )
        }
        
    @typecheck()
    def forward(self, z):
        #only use kwargs to pass things to neural modules
        #TODO: check should this be z=z?
        return self.decoder(z)

    def _step(self, batch, batch_idx):
        # TODO: need way to dynammically unpack batch
        data = batch
        data = data.view(data.shape[0], -1) #TODO: find out if this should always happen

        # get posterior estimate (get params that describe gaussian)
        x = self.encoder(data)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        std = torch.exp(log_var / 2)

        # get distributions TODO: maybe refactor
        # TODO later, parameterize distrubtion options
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # sample z, get reconstruction
        z = q.rsample()
        data_hat = self.forward(z=z)

        # Reconstruction Loss
        # TODO: parameterize recon loss function
        recon_loss = F.mse_loss(data_hat, data, reduction='mean')
        
        # KL Loss
        log_qz = q.log_prob(z) #consider naming changes
        log_pz = p.log_prob(z)
        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.cfg.kl_coeff

        # Total Loss
        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self._step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def setup_training_data(self, train_data_layer_config = None):
        """ Creates dataset, wrap it with dataloader and return the latter """
        # Instantiate dataset.
        mnist_ds = MNISTDataset(self._cfg.dataset)
        # Configure data loader.
        train_dataloader = DataLoader(dataset=mnist_ds, **(self._cfg.dataloader))
        self._train_dl = train_dataloader

    def setup_validation_data(self, val_data_layer_config = None):
        """ Not implemented. """
        self._val_dl = None	

    def training_epoch_end(self, outputs):
        #TODO: find out if this mean is correct
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_mean', train_loss_mean)

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x for x in outputs]).mean()
        self.log('val_loss_mean', val_loss_mean)

    @classmethod
    def list_available_models(cls):
        """ Not implemented. """
        pass


@dataclass
class MLPConf():
    _target_: str = 'nero.collections.common.mlp.MultiLayerPerceptron'
    hidden_size: int = MISSING
    num_classes: int = MISSING
    activation: str = MISSING
    log_softmax: bool = False

@dataclass
class LEGOModelConf():
    encoder: Any = MLPConf(
                           hidden_size=100,
                           num_classes=256,
                           activation='relu',
                           log_softmax=False
                          )
    decoder: Any = MLPConf(
                           hidden_size=2,
                           num_classes=100,
                           activation='relu',
                           log_softmax=False
                          )
    latent_dim: int = 2
    kl_coeff: float = 0.1

@dataclass
class LEGOConf():
    data_dir: Any = '/home/rosario/data/'
    data_date: Any = '2020-10-25'
    trainer: TrainerConf = TrainerConf()
    model: Any = LEGOModelConf()
    num_workers: int = 1
    batch_size: int = 16

cs = ConfigStore.instance()
cs.store(name="legoconf", node=LEGOConf)


@hydra.main(config_name='legoconf')
def main(cfg):
    from nero.collections.planning.datamodules.lego import LegoDataModule
    print(cfg.pretty())
    import ipdb; ipdb.set_trace()
    #ROOT_DIR = hydra.utils.get_original_cwd()
    
    dm = LegoDataModule(data_dir=cfg.data_dir,
                        data_date=cfg.data_date,
                        num_workers=cfg.num_workers,
                        batch_size=cfg.batch_size,
                       )
    
    model = LEGO(cfg.model)


    #TODO: hydra.instantiate
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
        
