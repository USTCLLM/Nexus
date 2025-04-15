from __future__ import absolute_import, division, print_function, unicode_literals
import os
import ctypes
import random
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import csv
import numpy.typing as npt
from cachetools import LRUCache

import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass
from torch.utils.data import DataLoader, IterableDataset, Dataset, get_worker_info
import torch.distributed as dist
from typing import List, Tuple, Union

from Nexus.abc.training.reranker import AbsRerankerCollator
from Nexus.training.reranker.recommendation.arguments import DataArguments
from Nexus.modules.dataset import get_client, process_conditions, smart_read_pl, get_pyarrow_format

def _to_tensor(array: pa.Array):
    """
    Convert pyarrow.Array to PyTorch Tensor。
    :param array: A pyarrow.Array type object
    :return: PyTorch Tensor
    """
    def numpy_to_tensor(array: npt.NDArray) -> torch.Tensor:
        if not array.flags.writeable:
            array = np.array(array)
        return torch.from_numpy(array)

    # check if there is null value
    if array.null_count > 0:
        raise ValueError("Array contains null values, which are not supported.")

    if pa.types.is_integer(array.type) or pa.types.is_floating(array.type):
        return numpy_to_tensor(array.to_numpy(zero_copy_only=False))
    
    elif pa.types.is_boolean(array.type):
        return numpy_to_tensor(array.to_numpy(zero_copy_only=False).astype(np.bool_))
    
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        # deal with ListArray
        if isinstance(array, pa.ChunkedArray):
            array = array.combine_chunks()
        assert isinstance(array, pa.ListArray) or isinstance(array, pa.LargeListArray)
        offsets = array.offsets.to_numpy()
        num_rows = len(offsets) - 1
        lengths = offsets[1:] - offsets[:-1]

        if not np.all(lengths == lengths[0]):
            raise ValueError("Not fixed-length list array.")

        list_size = lengths[0]


        values = array.values  # critical to keep reference alive
        dtype = values.type.to_pandas_dtype()
        itemsize = np.dtype(dtype).itemsize

        # Use memoryview for safer buffer management
        start = offsets[0] + values.offset
        end = offsets[-1] + values.offset
        buf = values.buffers()[1]  # actual data buffer

        mv = memoryview(buf)[start * itemsize : end * itemsize]
        np_array = np.frombuffer(mv, dtype=dtype).reshape(num_rows, list_size)

        # start = offsets[0]
        # end = offsets[-1]

        # # === Key Logic：Original arrays ===
        # values = array.values
        # buf = values.buffers()[1]
        # dtype = values.type.to_pandas_dtype()  # int32/int64/bool

        # # === Use ctypes to construct NumPy array (zero-copy + writable) ===
        # itemsize = np.dtype(dtype).itemsize
        # values_offset = values.offset
        # ptr = int(buf.address + (values_offset + start) * itemsize)
        # length = int((end - start) * itemsize)
        # raw = (ctypes.c_char * length).from_address(ptr)

        # np_array = np.frombuffer(raw, dtype=dtype).reshape(num_rows, list_size)
        # np_array.setflags(write=True)  # allow torch.from_numpy

        # test_np_array = np.stack(array.to_numpy(zero_copy_only=False))
        # if (test_np_array - np_array).sum() != 0:
        #     print("Error in converting list array to numpy array")
        return torch.from_numpy(np_array)
    
    else:
        raise NotImplementedError(f"Unsupported Arrow type: {array.type}")



class ItemDataset(Dataset):
    def __init__(self, item_feat_df: torch.Tensor):
        super().__init__()
        self.item_feat_df = item_feat_df
        self.item_ids = item_feat_df.index.to_numpy()
        self.item_id_2_offset = pd.DataFrame.from_dict(
            {item_id: i for i, item_id in enumerate(self.item_ids)},
            orient="index",
            columns=["offset"]
        )
        self.item_id_col = self.item_feat_df.index.name

    def __getitem__(self, index):
        feat_dict = self.item_feat_df.loc[self.item_ids[index]].to_dict()
        if self.item_id_col is not None:
            feat_dict.update({self.item_id_col: self.item_ids[index]})
        return feat_dict
    
    def __len__(self):
        return len(self.item_ids)
    

class ConfigProcessor(object):
    """
    This class is used to split the data into train and test sets at file level.

    Args:
        config (Union[dict, str]): Configuration dictionary or path to JSON file
    """
    def __init__(self, config: Union[dict, str, DataArguments]):
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
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)



class ShardedDatasetPA(IterableDataset):
    def __init__(self, config: DataArguments, batch_size=8192, shuffle=False, seed=42, seq_cache_max_size=1, **kwargs):
        super(ShardedDatasetPA, self).__init__()
        self.seed = seed
        random.seed(seed)
        self.config = config
        self.data_client = get_client(config.type, config.url)
        
        self.item_feat_dataset = None

        self.filenames = self.config.files
        if shuffle:
            random.shuffle(self.filenames)
        
        self._batch_size = batch_size
        self._shuffle_buffer_batches_num = 5
        self._shuffle = shuffle
        self._drop_remainder = True

        self._seq_feat_cache = LRUCache(maxsize=seq_cache_max_size)

        seq_join_keys = list(set([seq_info["key"] for seq_info in self.config.user_sequential_info]))
        
        self.selected_cols = self.config.context_features + self.config.item_features + self.config.labels
        self.selected_cols += seq_join_keys
        self.selected_cols = list(set(self.selected_cols))
        
    def __len__(self):
        return 10000000000
    
    def get_worker_info(self) -> Tuple[int, int]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        return rank * num_workers + worker_id, num_workers * world_size
    

    def get_feat_file_from_data_file(self, seq_info, data_file: str) -> str:
        return os.path.join(seq_info["url"], data_file)
    
    
    def _concat_seq_data(self, filename: str, batch_data):
        if len(self.config.user_sequential_info) <= 0:
            return batch_data
        src_file_base_name = os.path.basename(filename)
        if src_file_base_name not in self._seq_feat_cache:
            seq_dict = {}
            for i, seq_info in enumerate(self.config.user_sequential_info):
                seq_name = seq_info["name"]
                join_key = seq_info["key"]
                feat_file = self.get_feat_file_from_data_file(seq_info, src_file_base_name)
                new_col_names = [f"{seq_name}[nexus-sep]{col}" if col != join_key else col for col in seq_info["use_cols"]]
                seq_df = smart_read_pl(
                    file_path=feat_file,
                    columns=seq_info["use_cols"] + [join_key],
                ).rename(dict(zip(seq_info["use_cols"], new_col_names)))
                seq_dict[seq_name] = (join_key, seq_df)
            self._seq_feat_cache[src_file_base_name] = seq_dict
        else:
            seq_dict = self._seq_feat_cache[src_file_base_name]

        # convert batch_data to polars DataFrame  
        batch_df = pl.from_arrow(batch_data)
        
        # concat all sequential features
        join_keys = list(set([seq[0] for seq in seq_dict.values()]))
        for seq_name, seq in seq_dict.items():
            join_key, seq_df = seq
            batch_df = batch_df.join(
                seq_df,
                on=join_key,
                how="left",
            )
        
        # drop the join keys
        batch_df = batch_df.drop(join_keys)
        
        # convert batch_df to pyarrow Table
        data = batch_df.to_arrow()
        return data

    
    def _build_batch(self, table: pa.Table):
        batch_data = {}
        for k, v in zip(table.column_names, table.columns):
            if "[nexus-sep]" in k:
                # split the column name
                seq_name, col_name = k.split("[nexus-sep]")
                if seq_name not in batch_data:
                    batch_data[seq_name] = {}
                t = _to_tensor(v)
                # assert (t - torch.from_numpy(np.stack(v.to_numpy(zero_copy_only=False)))).sum() == 0, f"Error in converting {k} to tensor"
                batch_data[seq_name][col_name] = t
                # if col_name == "author_id":
                #     if batch_data[seq_name][col_name].max() >= 33474011:
                #         print("Bug!!!")
            else:
                batch_data[k] = _to_tensor(v)
        return batch_data
    
    def __iter__(self):
        worker_id, num_workers = self.get_worker_info()
        
        assert num_workers <= len(self.filenames), f"num_workers {num_workers} > num_files {len(self.filenames)}"
        cur_worker_filenames = [os.path.join(self.config.url, fname) for fname in self.filenames[worker_id::num_workers]]

        dataset = ds.dataset(
            cur_worker_filenames,
            format=get_pyarrow_format(cur_worker_filenames[0]),
        )

        scanner = dataset.scanner(batch_size=self._batch_size, columns=self.selected_cols).scan_batches()

        shuffle_buffer = []
        while True:
            data = None
            try:
                data = next(scanner)
            except StopIteration:
                data = None if self._drop_remainder else data
            
            if data is not None:
                source_file = data.fragment.path

                data = self._concat_seq_data(source_file, data.record_batch)

                if self._shuffle:
                    shuffle_buffer.append(data)
                    if len(shuffle_buffer) < self._shuffle_buffer_batches_num:
                        continue
                    else:
                        idx = random.randrange(len(shuffle_buffer))
                        data = shuffle_buffer.pop(idx)
                
                yield self._build_batch(data)

            if data is None:
                break

        if len(shuffle_buffer) > 0:
            random.shuffle(shuffle_buffer)
            for data in shuffle_buffer:
                yield self._build_batch(data)


                    
@dataclass
class AbsRecommenderRerankerCollator(AbsRerankerCollator):
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



if __name__ == "__main__":
    # config_path = "/share/project/huangxu/Nexus/examples/recommendation/config/data/recflow_ranker_local.json"
    config_path = "/share/project/huangxu/Nexus/examples/recommendation/config/data/recflow_ranker_local_seq.json"
    config = ConfigProcessor(config_path)

    train_config, eval_config = config.split_config()

    train_data = ShardedDatasetPA(train_config, shuffle=True)
    test_data = ShardedDatasetPA(eval_config, shuffle=False)

    from torch.utils.data import DataLoader
    import time

    dataloader = DataLoader(train_data, batch_size=None, num_workers=0, shuffle=False)
    tic = time.time()
    for i, data in enumerate(dataloader):
        if i % 100 == 0:
            print(i)
    toc = time.time()
    print(f"Done, time: {toc - tic} seconds")

