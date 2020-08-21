import os
import pickle
import numpy as np
import pandas as pd
from random import choices
from typing import Dict

import torch
from torch.utils.data.dataset import Dataset, TensorDataset

from utils import load_seperator_config


class SeparationDataset(Dataset):
    def __init__(self, config: Dict = None, combination_list_path: str = None):
        super().__init__()
        if config is None:
            config = load_seperator_config()
        self.human_set = self._convert_to_dataset(self._load_data(config, 'experiment'))
        self.clutter_set = self._convert_to_dataset(self._load_data(config, 'empty'))
        if combination_list_path is None:
            self.combination_list = self._create_combination_list(len(self.human_set), len(self.clutter_set), int(config['training']['dataset_length']))
        else:
            self.combination_list = self._load_combination_list(combination_list_path)

    @staticmethod
    def _load_combination_list(combination_list_path):
        with open(combination_list_path, 'rb') as f:
            combination_list = pickle.load(f)
        return combination_list

    def save_combination_list(self, combination_list_path):
        with open(combination_list_path, 'wb') as f:
            pickle.dump(self.combination_list, f)

    @staticmethod
    def _load_data(config: Dict, name: str) -> Dict:
        # Read data from files:
        set_name = config['datasets'][name][0]
        with open(os.path.join(config['folders']['data'], set_name + '.pkl'), 'rb') as f:
            data = pickle.load(f)
        meta = pd.read_csv(os.path.join(config['folders']['data'], set_name + '.csv'))
        # Unpack to dictionary and cast to numpy arrays:
        data_dictionary = {**meta, **data}
        for key, item in data_dictionary.items():
            data_dictionary[key] = np.array(item)
        return data_dictionary

    def _convert_to_dataset(self, data):
        max_log_power_tensor = torch.from_numpy(self._get_max_log_power(data['iq_sweep_burst']))
        iq_data_tensor = torch.from_numpy(data['iq_sweep_burst']).type(torch.cfloat)
        flat_data_tensor = torch.reshape(torch.view_as_real(iq_data_tensor.permute(0,2,1)),(-1,32*128*2))
        dataset = TensorDataset(flat_data_tensor, max_log_power_tensor)
        return dataset

    @staticmethod
    def _get_max_log_power(x):
        x = np.log(np.abs(np.fft.fft(x, n=128, axis=1)) + 1)
        x = x.reshape((-1, 128*32))
        return x.max(axis=1)

    @staticmethod
    def _create_combination_list(len_human: int, len_clutter: int, required_dataset_length):
        human_list = choices([*range(len_human)], k=required_dataset_length)
        clutter_list = choices(range(len_clutter), k=required_dataset_length)
        pairs = list(zip(human_list, clutter_list))
        return pairs

    def __len__(self):
        return len(self.combination_list)

    def __getitem__(self, index):
        human_data, human_power = self.human_set[self.combination_list[index][0]]  # noqa
        clutter_data, clutter_power = self.clutter_set[self.combination_list[index][1]]  # noqa
        return human_data + clutter_data, torch.stack((human_data, clutter_data))


if __name__ == "__main__":
    config = load_seperator_config()
    sep_dataset = SeparationDataset(config, 5e6)
    sep_dataset[0]  # noqa