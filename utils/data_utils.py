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
        
        reference_values = torch.tensor([random.choice([-value for value in nl if value < original_value] +
                                                     [value for value in nl if value < original_value] or [0])
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


