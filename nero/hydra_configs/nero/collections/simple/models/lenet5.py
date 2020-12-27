from dataclasses import dataclass
from typing import Any

from hydra_configs.torch.utils.data.dataset import DatasetConf
from hydra_configs.torch.utils.data.dataloader import DataLoaderConf
from hydra_configs.torch.optim import AdamConf
from hydra_configs.torch.nn.modules.loss import NLLLossConf

from hydra_configs.nero.collections.simple.datasets.mnist import MNISTDatasetConf

@dataclass
class MNISTLeNet5Conf():
    """
    Structured config for LeNet-5 model class - that also contains parameters of dataset and dataloader.
    """

    dataset: DatasetConf = MNISTDatasetConf(width=32, height=32)
    dataloader: DataLoaderConf = DataLoaderConf(batch_size=64, shuffle=True)
    module: Any = None
    loss: Any = NLLLossConf()
    optim: Any = AdamConf()

