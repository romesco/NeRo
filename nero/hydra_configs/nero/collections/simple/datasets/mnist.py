from dataclasses import dataclass
from hydra_configs.torch.utils.data.dataset import DatasetConf

@dataclass
class MNISTDatasetConf(DatasetConf):
    """
    Structured config for MNISTDataset class.
    Args:
        height: image height (DEFAULT: 28)
        width: image width (DEFAULT: 28)
        data_folder: path to the folder with data, can be relative to user (DEFAULT: "~/data/mnist")
        train: use train or test splits (DEFAULT: True)
        name: Name of the module (DEFAULT: None)
    """

    height: int = 28
    width: int = 28
    data_folder: str = "~/data/mnist"
    train: bool = True
    download: bool = True
