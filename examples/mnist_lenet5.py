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

import pytorch_lightning as pl
import hydra
from hydra.core.config_store import ConfigStore

from dataclasses import dataclass
from nero.collections.example.models.lenet5 import MNISTLeNet5Conf, MNISTLeNet5

from hydra_configs.pytorch_lightning.trainer import TrainerConf

@dataclass
class MNISTConf:
    model: MNISTLeNet5Conf = MNISTLeNet5Conf
    trainer: TrainerConf = TrainerConf(gpus=None)

cs = ConfigStore.instance()
cs.store(name="mnistconf", node=MNISTConf)

@hydra.main(config_name='mnistconf')    
def main(cfg):
    print(cfg.pretty())

    # The "model" - with dataloader/dataset inside of it.
    lenet5 = MNISTLeNet5(cfg.model)

    #lenet5.setup_training_data()
    # Setup optimizer and scheduler
    #lenet5.setup_optimization()

    # Create trainer.
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Train.
    trainer.fit(model=lenet5)


if __name__ == "__main__":
    main()
