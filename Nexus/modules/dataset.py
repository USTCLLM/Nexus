import json
from datetime import datetime
import torch
from typing import Dict, List
from copy import deepcopy
import re
import os 
import fsspec
import pandas as pd
import polars as pl
from pathlib import Path


class BaseClient(object):
    def __init__(self, url: str):
        self.url = url
        
    def build_file_index(self, file_partition_format: Dict[str, str]) -> dict:
        filenames = self.list_dir()
        extract_func = extract_number if file_partition_format['type'] == "number" else extract_timestamp
        dates_or_numbers = [extract_func(filename, file_partition_format["format"]) for filename in filenames]
        number_to_file = {}
        for i, filename in enumerate(filenames):
            number_to_file[dates_or_numbers[i]] = filename
        return number_to_file

    def load_file(self, path=None, **kwargs):
        if not path:
            path = self.url
        else:
            path = os.path.join(self.url, path)
        filetype = detect_file_type(path)
        if filetype == "parquet":
            df = pd.read_parquet(path, **kwargs)
        elif filetype == "csv":
            df = pd.read_csv(path, **kwargs)
        elif filetype == "feather":
            df = pd.read_feather(path, **kwargs)
        elif filetype == "pkl":
            df = pd.read_pickle(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {filetype}")
        return df
    
    def list_dir(self) -> list[str]:
        """List all files and directories in the given directory."""
        return sorted(os.listdir(self.url))
    
    def get_train_eval_filenames(
            self, 
            file_partition_format: Dict[str, str],
            train_period: Dict[str, str],
            eval_period: Dict[str, str],
        ):
        filenames = self.list_dir()
        extract_func = extract_number if file_partition_format["type"] == "number" else extract_timestamp
        name_format = file_partition_format["format"]
        dates_or_numbers = [extract_func(filename, file_partition_format["format"]) for filename in filenames]
        train_files = [filename for date_or_number, filename in zip(dates_or_numbers, filenames)
                       if date_or_number >= extract_func(train_period["start_date"], name_format)
                       and date_or_number < extract_func(train_period["end_date"], name_format)]
        eval_files = [filename for date_or_number, filename in zip(dates_or_numbers, filenames)
                       if date_or_number >= extract_func(eval_period["start_date"], name_format) 
                       and date_or_number < extract_func(eval_period["end_date"], name_format)]
        return train_files, eval_files
    
    
class HDFSClient(BaseClient):
    def __init__(self, url: str):
        self._check_hdfs_connection(url)
        super(HDFSClient, self).__init__(url)
        
    @staticmethod
    def _check_hdfs_connection(hdfs_url: str) -> bool:
        """Check that we can connect to the HDFS filesystem at url"""
        if not isinstance(hdfs_url, str):
            raise TypeError(f"Expected `url` to be a string, got {type(hdfs_url)}")
        if not hdfs_url.startswith("hdfs://"):
            raise ValueError(f"Expected `url` to start with 'hdfs://', got {hdfs_url}")
        try:
            fs = fsspec.filesystem('hdfs', fs_kwargs={'hdfs_connect': hdfs_url})
        except ImportError:
            raise ImportError("`fsspec` is not installed")
        except Exception as e:
            print(e)
            raise ValueError(f"Could not connect to {hdfs_url}")
        return True
    
    def list_dir(self) -> list[str]:
        """List all files and directories in the given directory."""
        fs = fsspec.filesystem('hdfs')
        return sorted([os.path.basename(file) for file in fs.ls(self.url)])
    
CLIENT_MAP = {
    'file': BaseClient,
    'hdfs': HDFSClient
}

def get_client(client_type: str, url: str):
    if client_type in CLIENT_MAP.keys():
        return CLIENT_MAP[client_type](url=url)
    else:
        raise ValueError(f"Unknown client type: {client_type}")
    
    

def read_json(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_yaml(filepath: str):
    try:
        import yaml
    except ImportError:
        raise ImportError("Please install PyYAML first.")
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def timeformat_to_regex(timeformat: str):
    return re.compile(re.escape(timeformat).replace("%Y", r"\d{4}")\
                                 .replace("%m", r"\d{2}")\
                                 .replace("%d", r"\d{2}"))

def extract_timestamp(filename: str, format: str="%Y-%m-%d") -> datetime:
    """Extract timestamp from filename using regex according to the given format"""
    # get regex according to the given format
    regex = timeformat_to_regex(format)
    match = re.search(regex, filename)
    if not match:
        raise ValueError(f"No date found in {filename}")
    return datetime.strptime(match.group(), format)


def extract_number(filename: str, format: str="{\d{4})") -> int:
    """Extract number from filename using regex according to the given format"""
    # get regex according to the given format
    match = re.search(format, filename)
    if not match:
        raise ValueError(f"No number found in {filename} using pattern '{format}'")
    return int(match.group(1))


def time2datestr(dt: datetime, format: str="%Y-%m-%d") -> str:
    return dt.strftime(format)


def nested_dict_update(dict_1: dict, dict_2: dict):
    """Update a nested dictionary (dict_1) with another one (dict_2)
    
    Args:
        dict_1 (dict): The target dictionary
        dict_2 (dict): The source dictionary

    Returns:
        dict: The updated dictionary
    """
    res = deepcopy(dict_1)
    for key, value in dict_2.items():
        if key not in res:
            res[key] = {}
        if isinstance(value, dict) and (res[key] is not None):
            res[key] = nested_dict_update(res[key], value)
        else:
            res[key] = value
    return res


def df_to_tensor_dict(df):
    """Convert pandas dataframe into tensor dictionary"""
    return {col: torch.tensor(df[col].values) for col in df.columns}


def process_conditions(conditions: List[str]):
    """
    Process conditions list. From str such as "==3" to a lambda function.
    Supported operators: [==, !=, >, <, >=, <=]
    Args:
        conditions (List[str]): A list of conditions
    Returns:
        function: A lambda function covering all conditions
    """
    def parse_condition(condition_str):
        # Define a regular expression to match the operator and value in the condition string
        match = re.match(r'(==|!=|>=|<=|>|<)(\d+)', condition_str.strip().replace(" ", ""))
        if not match:
            raise ValueError(f"Unsupported condition format: {condition_str}")
        
        operator, value = match.groups()
        value = int(value)  # 将数值转换为整数
        
        # Return a lambda function based on the operator
        if operator == '==':
            return lambda x: x == value
        elif operator == '!=':
            return lambda x: x != value
        elif operator == '>=':
            return lambda x: x >= value
        elif operator == '<=':
            return lambda x: x <= value
        elif operator == '>':
            return lambda x: x > value
        elif operator == '<':
            return lambda x: x < value
        else:
            raise ValueError(f"Unsupported operator in condition: {operator}")

    # Convert each condition string to a lambda function and combine them with logical AND
    lambda_functions = [parse_condition(cond) for cond in conditions]
    return lambda_functions


def detect_file_type(path: str) -> str:
    """Detect the file type from its extension"""
    file_extension = path.split('.')[-1]
    if file_extension == "parquet":
        return "parquet"
    elif file_extension == "feather":
        return "feather"
    elif file_extension in ["csv", "txt"]:
        return "csv"
    elif file_extension in ["pkl", "pickle"]:
        return "pkl"
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    

def smart_read_pl(file_path, **kwargs):
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == ".parquet":
        return pl.read_parquet(file_path, **kwargs)
    elif file_ext == ".csv":
        return pl.read_csv(file_path, **kwargs)
    elif file_ext == ".json":
        return pl.read_json(file_path, **kwargs)
    elif file_ext == ".ipc" or file_ext == ".arrow":
        return pl.read_ipc(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    

def get_pyarrow_format(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".parquet":
        return "parquet"
    elif ext == ".csv":
        return "csv"
    elif ext == ".json":
        return "json"
    elif ext == ".ipc" or ext == ".arrow":
        return "ipc"
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    
