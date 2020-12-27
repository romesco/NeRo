import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

 # inherits from torch Dataset class
class LegoDataset(Dataset):

    def __init__(self, occgrid_data, start_goal_data, bottleneck_data):
        super().__init__()
        self.occgrid_data = occgrid_data
        self.start_goal_data = start_goal_data
        self.bottleneck_data = bottleneck_data
        self.length = len(self.occgrid_data)
        # Check that all the three matrices have same length.
        
    def __getitem__(self, index):
    	# convert to float32 tensor
        occgrid = torch.tensor(self.occgrid_data[index]).float()
        #TEMP HACK:
        occgrid = occgrid.unsqueeze(0)
        start_goal = torch.tensor(self.start_goal_data[index]).float()
        bottleneck = torch.tensor(self.bottleneck_data[index]).float()
        return occgrid #, start_goal, bottleneck
    
    def __len__(self):
        return self.length

class LegoDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir: str,
            data_date: str,
            num_workers: int,
            batch_size: int,
    ):
        super().__init__()
        self.val_percent = 0.05
        self.data_dir = data_dir
        self.data_date = data_date
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.occgrid_size = (1, 10,10)
        self.start_goal_size = (2,2)
        self.bottleneck_size = (1,2)

    def setup(self, stage=None):
        # Specify the paths to the data.
        occgrid_filepath = os.path.join(self.data_dir, 'LEGO/', self.data_date, 'occupancy_grid.npy')
        start_goal_filepath = os.path.join(self.data_dir, 'LEGO/', self.data_date, 'start_goal.npy')
        bottleneck_filepath = os.path.join(self.data_dir, 'LEGO/', self.data_date, 'bottleneck.npy')

        # Load into numpy arrays.
        occgrids = np.load(occgrid_filepath, allow_pickle=True)
        start_goals = np.load(start_goal_filepath, allow_pickle=True)
        bottlenecks = np.load(bottleneck_filepath, allow_pickle=True)

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            dataset_full = LegoDataset(occgrids, start_goals, bottlenecks)
            val_len = int(self.val_percent*len(dataset_full))
            train_len = len(dataset_full) - val_len 
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [train_len, val_len])
                

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          )

if __name__ == '__main__':
	lego_module = LegoDataModule()
	lego_train = lego_module.train_dataloader()
	import ipdb; ipdb.set_trace()
