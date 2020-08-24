import os
import pickle
import numpy as np
import pandas as pd
from random import choices
from typing import Dict, List

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.dataset import Dataset, TensorDataset

from ..utils import load_classifier_config


class MAFATDataset(Dataset):
    def __init__(self, name_list: List[str], config: Dict = None):
        super().__init__()
        if config is None:
            config = load_classifier_config()
        dataset_list = []
        target_segment_ids_list = []
        target_geolocation_type_list = []
        target_type_list = []
        for name in name_list:
            target_data = self._load_dataset(config, name)
            # Load human \ animal dataset:
            dataset_list.append(self._convert_to_dataset(target_data))
            target_segment_ids_list.append(target_data['segment_id'])
            target_geolocation_type_list.append(target_data['geolocation_type'] if 'geolocation_type' in target_data.keys() else None)
            target_type_list.append(target_data['target_type'] if 'target_type' in target_data.keys() else None)
        self.target_set = ConcatDataset(dataset_list)
        self.target_segment_ids = np.concatenate(target_segment_ids_list, axis=0)
        self.target_geolocation_type = np.concatenate(target_geolocation_type_list, axis=0) if target_geolocation_type_list[0] is not None else None
        self.target_type = np.concatenate(target_type_list, axis=0) if target_type_list[0] is not None else None

    @staticmethod
    def _load_dataset(config, name) -> Dict:
        # Read data from files:
        set_name = config['datasets'][name][0]
        n_max = config['datasets'][name][1]
        with open(os.path.join(config['folders']['data'], set_name + '.pkl'), 'rb') as f:
            data = pickle.load(f)
        meta = pd.read_csv(os.path.join(config['folders']['data'], set_name + '.csv'))
        # Unpack to dictionary and cast to numpy arrays:
        data_dict = {**meta, **data}
        for key, item in data_dict.items():
            data_dict[key] = np.array(item)
        # Restrict data set size:
        n_set = len(data_dict['segment_id'])
        if n_max != -1 and n_set > n_max:
            sample_indexes = np.random.choice(n_set, size=n_max, replace=False)
            for key, item in data_dict.items():
                data_dict[key] = item[sample_indexes] if item.ndim == 1 else item[sample_indexes, :] if item.ndim == 2 else item[sample_indexes, :, :]
        return data_dict

    def _get_labels(self, data: Dict) -> List:
        if 'target_type' in data.keys():
            labels = [1 if label == "human" else 0 if label == "animal" else 2 for label in data["target_type"]]
        else:
            labels = [-1 for i in range(len(data['segment_id']))]
        return labels

    def _convert_to_dataset(self, data):
        iq_data_tensor = torch.from_numpy(data['iq_sweep_burst']).type(torch.cfloat)
        labels_tensor = torch.tensor(self._get_labels(data), dtype=torch.long, requires_grad=False)
        dataset = TensorDataset(iq_data_tensor, labels_tensor)
        return dataset

    def __len__(self):
        return len(self.target_set)

    def __getitem__(self, index):
        return self.target_set[index]


class MAFATDatasetAugmented(MAFATDataset):
    def __init__(self, name_list: List[str], config:Dict = None, combination_list_path: str = None, n_augmentations: int = 2):
        super().__init__(name_list=name_list, config=config)
        # Load empty dataset:
        background_data = self._load_dataset(config, 'empty')
        background_data = self.add_empty_background(background_data)
        self.background_set = self._convert_to_dataset(background_data)
        self.background_segment_ids = background_data['segment_id']
        self.background_geolocation_type = background_data['geolocation_type']
        # Create combination list:
        if combination_list_path is None:
            self.combination_list = self._create_combination_list(len(self.target_set), len(self.background_set), n_augmentations)
        else:
            self.combination_list = self._load_combination_list(combination_list_path)

    def add_empty_background(self, data):
        data['segment_id'] = np.insert(data['segment_id'], 0, -1, axis=0)
        data['track_id'] = np.insert(data['track_id'], 0, -1, axis=0)
        data['geolocation_type'] = np.insert(data['geolocation_type'], 0, 'Z', axis=0)
        data['geolocation_id'] = np.insert(data['geolocation_id'], 0, -1, axis=0)
        data['sensor_id'] = np.insert(data['sensor_id'], 0, -1, axis=0)
        data['snr_type'] = np.insert(data['snr_type'], 0, 'EMPTY', axis=0)
        data['date_index'] = np.insert(data['date_index'], 0, -1, axis=0)
        data['target_type'] = np.insert(data['target_type'], 0, 'empty', axis=0)
        data['doppler_burst'] = np.insert(data['doppler_burst'], 0, np.zeros((1, 32)), axis=0)
        data['iq_sweep_burst'] = np.insert(data['iq_sweep_burst'], 0, np.zeros((1, 128, 32)), axis=0)
        return data


    @staticmethod
    def _load_combination_list(combination_list_path):
        with open(combination_list_path, 'rb') as f:
            combination_list = pickle.load(f)
        return combination_list

    def save_combination_list(self, combination_list_path):
        with open(combination_list_path, 'wb') as f:
            pickle.dump(self.combination_list, f)

    @staticmethod
    def _create_combination_list(len_target: int, len_background: int, n_augmentations):
        target_list = np.repeat(np.arange(len_target), n_augmentations + 1, axis=0)
        background_list = np.array(choices(np.arange(1, len_background+1), k=n_augmentations * len_target))
        background_list = np.ravel(np.pad(np.reshape(background_list, (len_target, n_augmentations)), ((0, 0), (1, 0)))).tolist()
        pairs = list(zip(target_list, background_list))
        return pairs

    def __len__(self):
        return len(self.combination_list)

    def __getitem__(self, index):
        target_data, target_label = self.target_set[self.combination_list[index][0]]
        background_data, _ = self.background_set[self.combination_list[index][1]]
        return target_data + background_data, target_label
