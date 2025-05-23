from __future__ import absolute_import, division, print_function, unicode_literals
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from torch.utils.data import DataLoader, IterableDataset, Dataset


from Nexus.abc.training.embedder import AbsEmbedderCollator
from Nexus.training.embedder.recommendation.arguments import DataArguments
from Nexus.training.reranker.recommendation.dataset import ShardedDatasetPA as ShardedDatasetPA4Reranker
from Nexus.modules.dataset import get_client, process_conditions


def _load_popularity_dict(path: str) -> Dict[str, int]:
    if path.endswith(".json"):
        import json
        with open(path, "r") as f:
            item_pop_dict = json.load(f)
    elif path.endswith(".pkl"):
        import pickle
        with open(path, "rb") as f:
            item_pop_dict = pickle.load(f)
    elif path.endswith(".csv"):
        item_pop_dict = pd.read_csv(path, index_col=0).to_dict()
    elif path.endswith(".parquet"):
        item_pop_dict = pd.read_parquet(path, index_col=0).to_dict()
    elif path.endswith(".feather"):
        item_pop_dict = pd.read_feather(path, index_col=0).to_dict()
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return item_pop_dict


class ItemDataset(Dataset):
    def __init__(
            self,
            item_feat_df: pd.DataFrame,
            item_pop_dict_path = None,
        ):
        """
        Args:
            item_feat_df (pd.DataFrame): DataFrame containing item features
            item_pop_dict_path (str): Path to the item popularity dictionary
        """
        super().__init__()
        self.item_feat_df = item_feat_df
        self.item_ids = item_feat_df.index.to_numpy()
        self.item_id_2_offset = pd.DataFrame.from_dict(
            {item_id: i for i, item_id in enumerate(self.item_ids)},
            orient="index",
            columns=["offset"]
        )
        self.item_id_col = self.item_feat_df.index.name

        if item_pop_dict_path is not None:
            self.item_pop_dict = _load_popularity_dict(item_pop_dict_path)
            self.item_pop = np.array([self.item_pop_dict.get(item_id, 0) for item_id in self.item_ids])
        else:
            self.item_pop_dict = None
            self.item_pop = None

    def __getitem__(self, index: Union[int, torch.Tensor]):
        if isinstance(index, int):
            feat_dict = self.item_feat_df.loc[self.item_ids[index]].to_dict()
            if self.item_id_col is not None:
                feat_dict.update({self.item_id_col: self.item_ids[index]})
            return feat_dict
        elif isinstance(index, torch.Tensor):
            item_ids_list = index.view(-1).tolist()
            item_ids_list = self.item_ids[item_ids_list]
            feats = self.item_feat_df.loc[item_ids_list].to_dict("list")
            feats = {k: torch.tensor(v).reshape(*index.shape).to(index.device) for k,v in feats.items()}
            return feats
        else:
            raise ValueError("Index must be an integer or a tensor.")
        
    
    def __len__(self):
        return len(self.item_ids)
    

class ConfigProcessor(object):
    """
    This class is used to split the data into train and test sets at file level.

    Args:
        config (Union[dict, str]): Configuration dictionary or path to JSON file
    """
    def __init__(self, config: Union[dict, str]):
        self.config: DataArguments = self._load_config(config)

    def _load_config(self, config: Union[dict, str]) -> DataArguments:
        """Loads the configuration from either a dictionary or a JSON file."""
        if isinstance(config, DataArguments):
            args = config
        elif isinstance(config, dict):
            args = DataArguments.from_dict(config)
        elif isinstance(config, str):
            args = DataArguments.from_json(config)
        else:
            raise TypeError("Config must be a dictionary or path to JSON file.")
        return args

    def split_config(self) -> Tuple[DataArguments, DataArguments]:
        data_client = get_client(self.config.type, self.config.url)
        train_files, eval_files = data_client.get_train_eval_filenames(
            self.config.file_partition,
            self.config.train_period,
            self.config.test_period,
        )
        train_config = deepcopy(self.config)
        eval_config = deepcopy(self.config)
        train_config.files = train_files
        eval_config.files = eval_files
        return train_config, eval_config


class ShardedDataIterator(object):
    """This class is used to iterate over each day of data (dataframe)
    
    Args:
        config (dict): config dictionary
        filenames (List[str]): list of filenames to iterate over
    """
    def __init__(self, config: DataArguments):
        self.config = config
        self.data_client = get_client(config.type, config.url)
        self.number_to_filename = self.data_client.build_file_index(self.config.file_partition)
        self.filename_to_number = {filename: number for number, filename in self.number_to_filename.items()}
        self.seq_data_clients = []
        self.seq_number_to_filename = []
        if len(config.user_sequential_info) > 0:
            # use given key column and sequence data file
            for seq_info in config.user_sequential_info:
                client = get_client(config.type, seq_info["url"])
                self.seq_data_clients.append(client)
                self.seq_number_to_filename.append(client.build_file_index(config.file_partition))

        if config.item_info is not None:
            self.item_col = config.item_info["key"]
            self.item_data_client = get_client(config.type, config.item_info["url"])
        else:
            self.item_col = None
            self.item_data_client = None

        self.filenames = config.files
        self.cur_idx = 0
        data, seq_data = self.load_single_sharded_data()
        self._data_cache = {'filename': self.filenames[self.cur_idx], 'data': data, 'seq': seq_data}

    def _columns_rename(self, df: pd.DataFrame):
        """ Rename columns to remove whitespace and modify `features` in the config consistently"""
        rename_map = {}
        for col in df.columns:
            new_col = col.strip()
            rename_map[col] = new_col
        return df.rename(columns=rename_map)

    def load_file(self, filename: str):
        """Load one day of data"""
        data = self.data_client.load_file(filename)
        data = self._columns_rename(data)
        # use given columns
        # if self.config.context_features:
        seq_index_cols = [seq_info["key"] for seq_info in self.config.user_sequential_info]
        features = self.config.context_features + self.config.item_features + self.config.labels
        keep_cols = seq_index_cols + features
        data = data[keep_cols]
        data = self.post_process(data)
        return data
    
    
    def load_sequence_files(self, filename: str) -> List[pd.DataFrame]:
        """Load one day of sequence data"""
        if len(self.seq_data_clients) <= 0:
            return None
        date_or_number = self.filename_to_number[filename]
        seq_data_list = []
        for i, seq_data_client in enumerate(self.seq_data_clients):
            seq_info = self.config.user_sequential_info[i]
            seq_index_col = seq_info["key"]
            number_to_filename = self.seq_number_to_filename[i]
            seq_filename = number_to_filename[date_or_number]
            seq_data_i = seq_data_client.load_file(seq_filename)

            # sequence data is can be DataFrame or Dict
            if isinstance(seq_data_i, pd.DataFrame):
                seq_data_i.set_index(seq_index_col, inplace=True, drop=True)
            elif isinstance(seq_data_i, dict):
                # L*N -> n number of L
                seq_data_i = {k: [v[:, i] for i in range(v.shape[-1])] for k, v in seq_data_i.items()}
                seq_data_i = pd.DataFrame.from_dict(seq_data_i, orient='index', columns=seq_info["columns"])
            else:
                raise ValueError("Sequence data must be DataFrame or Dict")
            if seq_info["use_cols"] is not None:
                seq_data_i = seq_data_i[seq_info["use_cols"]]
            seq_data_list.append(seq_data_i.sort_index())
        return seq_data_list

    def load_item_file(self) -> pd.DataFrame:
        """Load all item data"""
        if self.config.item_info is None:
            return None
        data = self.item_data_client.load_file()
        if isinstance(data, pd.DataFrame):
            data.set_index(self.config.item_info["key"], inplace=True, drop=True)
        elif isinstance(data, dict):
            data = pd.DataFrame.from_dict(data, orient='index', columns=self.config.item_info["columns"])
        else:
            raise ValueError("Item data must be DataFrame or Dict")
        if self.config.item_info["use_cols"] is not None:
            data = data[self.config.item_info["use_cols"]]
        return data
        
    
    def post_process(self, df) -> pd.DataFrame:
        # filtering data in log table
        # case1: for retrieval model training
        if (self.config.filter_settings is not None) and (len(self.config.filter_settings) > 0):
            for col, conditions in self.config.filter_settings.items():
                condition_funcs = process_conditions(conditions)
                # df = df.loc[df[setting.by].map(condition_func)]
                for func in condition_funcs:
                    df = df[func(df[col])]
        return df


    def load_single_sharded_data(self):
        filename = self.filenames[self.cur_idx]
        return self.load_file(filename), self.load_sequence_files(filename)

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        self.cur_idx = 0
        return self

    def __next__(self):
        if self.cur_idx < len(self.filenames):
            cache_filename = self._data_cache.get('filename', None)
            if cache_filename is not None and self.filenames[self.cur_idx] == cache_filename:
                print(f"Load dataset file {self.filenames[self.cur_idx]} from cache")
                data, seq_data_list = self._data_cache['data'], self._data_cache['seq']
                print(f"Load dataset sucessfully.")
            else:
                print(f"Load dataset file {self.filenames[self.cur_idx]} from source")
                data, seq_data_list = self.load_single_sharded_data()
                print(f"Load dataset sucessfully.")
                self._data_cache['filename'] = self.filenames[self.cur_idx]
                self._data_cache['data'] = data
                self._data_cache['seq'] = seq_data_list
            self.cur_idx += 1
            return data, seq_data_list
        else:
            raise StopIteration
        


class ShardedDataset(IterableDataset):
    def __init__(self, config: DataArguments, shuffle=False, seed=42, **kwargs):
        super(ShardedDataset, self).__init__()
        self.seed = seed
        self.sharded_data_iterator = ShardedDataIterator(config)
        self.config = config
        self.seq_index_cols = [seq_info['key'] for seq_info in self.config.user_sequential_info]
        self.seq_data_names = [seq_info['name'] for seq_info in self.config.user_sequential_info]
        self.shuffle = shuffle
        item_data = self.sharded_data_iterator.load_item_file()
        self.item_feat_dataset = ItemDataset(item_data) if item_data is not None else None


    def __len__(self):
        return 1000000000

    def __iter__(self):
        self.seed += 1
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        for log_df, seq_df_list in self.sharded_data_iterator:
            if self.shuffle:
                log_df = log_df.sample(frac=1).reset_index(drop=True)
            log_dict = log_df.to_dict(orient='list')
            num_samples = log_df.shape[0]
            
            for i in range(num_samples):
                if i % num_workers == worker_id:
                    data_dict = {k: v[i] for k, v in log_dict.items()}
                    if len(seq_df_list) > 0:
                        for i, seq_df in enumerate(seq_df_list):
                            seq_data_dict = seq_df.loc[data_dict[self.seq_index_cols[i]]].to_dict()
                            seq_data_dict = {k: np.copy(v) for k, v in seq_data_dict.items()}
                            data_dict[self.seq_data_names[i]] = seq_data_dict
                    for index_key in self.seq_index_cols:
                        del data_dict[index_key]
                    yield (data_dict, )

    
    def get_item_loader(self, batch_size, num_workers=0, shuffle=False):
        return DataLoader(self.item_feat_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)



class ShardedDatasetPA(ShardedDatasetPA4Reranker):
    def __init__(self, config, batch_size=8192, shuffle=False, seed=42, seq_cache_max_size=1, **kwargs):
        super().__init__(config, batch_size, shuffle, seed, seq_cache_max_size, **kwargs)
        self.item_feat_dataset = self._load_item_data()


    def _load_item_data(self):
        if self.config.item_info is None:
            return None
        
        self.item_col = self.config.item_info["key"]
        self.item_data_client = get_client(self.config.type, self.config.item_info["url"])
        
        data = self.item_data_client.load_file()
        
        if isinstance(data, pd.DataFrame):
            data.set_index(self.item_col , inplace=True, drop=False)
        elif isinstance(data, dict):
            data = pd.DataFrame.from_dict(data, orient='index', columns=self.config.item_info["columns"])
        else:
            raise ValueError("Item data must be DataFrame or Dict")
        
        if self.config.item_info["use_cols"] is not None:
            data = data[self.config.item_info["use_cols"]]

        return ItemDataset(data, item_pop_dict_path=self.config.item_pop_dict_path)

    def get_item_loader(self, batch_size, num_workers=0, shuffle=False):
        return DataLoader(self.item_feat_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


                    
                    
@dataclass
class AbsRecommenderEmbedderCollator(AbsEmbedderCollator):
    """
    The abstract reranker collator.
    """    
    def __call__(self, features):
        features = [f[0] for f in features]
        all_dicts = {}
        
        for k, v in features[0].items():
            if isinstance(v, dict):
                all_dicts[k] = {}
                for _k in v.keys():
                    all_dicts[k][_k] = []
            else:
                all_dicts[k] = []
                
        for data_dict in features:
            for k, v in data_dict.items():
                if isinstance(v, dict):
                    for _k, _v in v.items():
                        if not isinstance(_v, torch.Tensor):
                            all_dicts[k][_k].append(torch.tensor(_v))
                        else:
                            all_dicts[k][_k].append(_v)
                else:
                    if not isinstance(v, torch.Tensor):
                        all_dicts[k].append(torch.tensor(v))
                    else:
                        all_dicts[k].append(v)
                
        for k, v in features[0].items():
            if isinstance(v, dict):
                for _k, _v in v.items():
                    all_dicts[k][_k] = torch.stack(all_dicts[k][_k], dim=0)
            else:
                all_dicts[k] = torch.stack(all_dicts[k], dim=0)
        return all_dicts
        # return features
        
def get_datasets(config: Union[dict, str]):
    config_processor = ConfigProcessor(config)
    train_config, eval_config = config_processor.split_config()

    train_data = ShardedDataset(train_config, shuffle=True, preload=False)
    test_data = ShardedDataset(eval_config, shuffle=False, preload=False)
    attr = train_config.to_attr()
    if train_data.item_feat_dataset is not None:
        # when candidate item dataset is given, the number of items is set to the number of items in the dataset
        # instead of the max item id in the dataset
        attr.num_items = len(train_data.item_feat_dataset)
    return (train_data, test_data), attr
