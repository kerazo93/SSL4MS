import os
import torch
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import lightning as L

class Trim(object):
    def __init__(self, max_len):
        assert isinstance(max_len, int)
        self.max_len = max_len
        
    def __call__(self, example):
        assert isinstance(example, pd.DataFrame)
        
        if example.shape[0]>self.max_len:
            ms1 = pd.DataFrame(example.iloc[:1])
            ms2 = example.iloc[1:]
            
            ms2 = ms2.sort_values('intensity', ascending=False)
            ms2 = ms2[:self.max_len-1]
            ms2 = ms2.sort_values('mz', ascending=False)
            
            example = pd.concat([ms1, ms2], ignore_index=True)
            
        return example
    

class Corrupt(object):
    def __init__(self, nl, prob=0.5):
        assert isinstance(nl, list)
        assert isinstance(prob, float)
        self.nl = nl
        self.prob = prob
        
    def __call__(self, example):
        assert isinstance(example, pd.DataFrame)
        
        mz = torch.Tensor(example['mz'])
        ints = torch.Tensor(example['intensity'])
        
        num_values_to_corrupt = int(len(mz) * self.prob)
        
        mz_idx_corrupt = random.sample(range(len(mz)), num_values_to_corrupt)
        ints_idx_corrupt = random.sample(range(len(ints)), num_values_to_corrupt)
        
        reference_values = torch.tensor([random.choice([-value for value in self.nl if value < original_value] +
                                                     [value for value in self.nl if value < original_value] or [0])
                                     for original_value in mz], dtype=mz.dtype)
        
        mz[mz_idx_corrupt] += reference_values[mz_idx_corrupt]
        
        original_sum = ints[ints_idx_corrupt].sum().item()
        
        replacement_values = torch.rand(num_values_to_corrupt)
        replacement_values = replacement_values / replacement_values.sum() * original_sum
        
        ints[ints_idx_corrupt] = replacement_values

        return torch.cat((mz.unsqueeze(1), ints.unsqueeze(1)), dim=1)
        

class Pad(object):
    def __init__(self, max_len):
        assert isinstance(max_len, int)
        self.max_len = max_len
        
    def __call__(self, example):
        assert isinstance(example, torch.Tensor)
        
        if example.shape[0]>=self.max_len:
            return example
            
        else:
            pad_rows = self.max_len - example.shape[0]

            padding_tensor = torch.zeros(pad_rows, example.shape[1], dtype=example.dtype)

            padded_tensor = torch.cat((example, padding_tensor), dim=0)

            return padded_tensor


class MassSpectraSelfSupervisedDataset(Dataset):
    def __init__(self, root_dir, csv_file, nl_file, corrupt_prob, max_len):
        self.root_dir = root_dir # where all the files are
        self.csv_file = csv_file # file that contains train/val/test split & labels
        self.nl_file = nl_file # csv file with neutral losses
        self.corrupt_prob = corrupt_prob # probability of data value getting corrupted
        self.max_len = max_len # max length of spectrum sequence
        
        self.trim = Trim(self.max_len) # trim transform
        self.nl = pd.read_csv(self.nl_file).values.flatten().tolist() #list of neutral losses
        self.corrupt = Corrupt(nl=self.nl, prob=self.corrupt_prob) # corrupting transform
        self.pad = Pad(self.max_len) # padding transform
        
        
    def __len__(self):
        return len(pd.read_csv(self.csv_file))
    
    def __getitem__(self, idx):
        files = pd.read_csv(self.csv_file)
        file = files
        
        path = os.path.join(self.root_dir, file.iloc[idx, 0]+'.csv')
        x = pd.read_csv(path)
        
        x_tilde = self.pad(self.corrupt(self.trim(x)))
        
        x_true = self.pad(torch.Tensor(self.trim(x).values))
        
        return x_tilde, x_true
    

class MassSpecSelfSupervisedDataModule(L.LightningDataModule):
    def __init__(self, root_dir, train_csv, val_csv, test_csv, nl_csv, corrupt_prob, max_len, batch_size):
        super().__init__()
        self.root_dir = root_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.nl_csv = nl_csv
        self.corrupt_prob = corrupt_prob
        self.max_len = max_len
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = MassSpectraSelfSupervisedDataset(root_dir=self.root_dir,
                                csv_file=self.train_csv,
                                nl_file=self.nl_csv,
                                corrupt_prob=self.corrupt_prob,
                                max_len=self.max_len)
            
        if stage == "validate" or stage is None:
            self.val_ds = MassSpectraSelfSupervisedDataset(root_dir=self.root_dir,
                                csv_file=self.val_csv,
                                nl_file=self.nl_csv,
                                corrupt_prob=self.corrupt_prob,
                                max_len=self.max_len)
        
        if stage == "test" or stage is None:
            self.test_ds = MassSpectraSelfSupervisedDataset(root_dir=self.root_dir,
                                csv_file=self.test_csv,
                                nl_file=self.nl_csv,
                                corrupt_prob=self.corrupt_prob,
                                max_len=self.max_len)
            
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)