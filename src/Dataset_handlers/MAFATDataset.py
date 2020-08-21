import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List
from itertools import product

import torch
from torch.utils.data.dataset import Dataset, TensorDataset


class MAFATDataset(Dataset):
    def __init__(self, config: Dict, name: str):
        super().__init__()
        target_types = ['empty', 'animal', 'human']
        geolocation_type = ['A', 'C', 'D']
        combinations = list(product(target_types, geolocation_type))
        self.idx_to_label = {ii: x for ii,x in enumerate(combinations)}
        self.label_to_idx = {x: ii for ii,x in enumerate(combinations)}
        data = self._load_dataset(config, name)
        self.segments_dataset = self._convert_to_dataset(data)
        self.segment_ids = data['segment_id']
        self.geolocation_type = data['geolocation_type'] if 'geolocation_type' in data.keys() else None
        self.target_type = data['target_type'] if 'target_type' in data.keys() else None

    @staticmethod
    def _load_dataset(config, name) -> Dict:
        # Read data from files:
        set_name = config['datasets'][name][0]
        n_max = config['datasets'][name][1]
        with open(os.path.join(config['folders']['data'], set_name + '.pkl'), 'rb') as f:
            data = pickle.load(f)
        meta = pd.read_csv(os.path.join(config['folders']['data'], set_name + '.csv'))
        # Unpack to dictionary and cast to numpy arrays:
        data_dictionary = {**meta, **data}
        for key, item in data_dictionary.items():
            data_dictionary[key] = np.array(item)
        # Restrict data set size:
        n_set = len(data_dictionary['segment_id'])
        if n_max != -1 and n_set > n_max:
            sample_indexes = np.random.choice(n_set, size=n_max, replace=False)
            for key, item in data_dictionary.items():
                data_dictionary[key] = item[sample_indexes] if item.ndim == 1 else item[sample_indexes, :] if item.ndim == 2 else item[sample_indexes, :, :]
        return data_dictionary

    def _get_labels(self, data: Dict) -> List:
        if 'target_type' in data.keys():
            # labels = [1 if label == "human" else 0 if label == "animal" else 2 for label in data["target_type"]]
            # labels = [0 if label == "A" else 1 if label == "C" else 2 for label in data["geolocation_type"]]
            labels = [self.label_to_idx[(target, geo)] for target, geo in zip(data["target_type"], data["geolocation_type"])]
        else:
            labels = [-1 for i in range(len(data['segment_id']))]
        return labels

    def _convert_to_dataset(self, data):
        iq_data_tensor = torch.from_numpy(data['iq_sweep_burst']).type(torch.cfloat)
        labels_tensor = torch.tensor(self._get_labels(data), dtype=torch.long, requires_grad=False)
        dataset = TensorDataset(iq_data_tensor, labels_tensor)
        return dataset

    def __len__(self):
        return len(self.segments_dataset)

    def __getitem__(self, index):
        iq_data, label = self.segments_dataset[index]
        return iq_data, label


class MAFATDatasetTwoHeads(MAFATDataset):
    def _get_labels(self, data: Dict) -> List:
        if 'target_type' in data.keys():
            labels = [1 if label == "human" else 0 for label in data["target_type"]]
        else:
            labels = [-1 for i in range(len(data['segment_id']))]
        return labels