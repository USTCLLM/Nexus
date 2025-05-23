
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from Nexus.abc.training.embedder import AbsEmbedderModelArguments, AbsEmbedderDataArguments, AbsEmbedderTrainingArguments
from typing import Dict
from Nexus.modules.arguments import DataAttr4Model, Statistics

@dataclass
class TrainingArguments(AbsEmbedderTrainingArguments):
    train_batch_size: int = 512

    cutoffs: int = field(default_factory=lambda : [1, 5, 10], metadata={"nargs": "+"})
    metrics: str = field(default_factory=lambda : ["ndcg", "recall"], metadata={"nargs": "+"})
    
    checkpoint_best_ckpt: bool = True   # if true, save best model in earystop callback
    checkpoint_steps: int = 1000    # if none, save model per epoch; else save model by steps
    
    optimizer: str = "adam"
    

@dataclass
class ModelArguments(AbsEmbedderModelArguments):
    # model_name: str = None
    # embedding_dim: int = 10
    data_config: DataAttr4Model = None
    embedding_dim: int = 10
    combined_embeddings: bool = False
    mlp_layers: int = field(default=None, metadata={"nargs": "+"})
    num_neg: int = 50
    activation: str = "relu"
    dropout: float = 0.3
    batch_norm: bool = True
    model_name_or_path: str = ''
    
    @staticmethod
    def from_dict(d: dict):
        arg = ModelArguments()
        for k, v in d.items():
            setattr(arg, k, v)
        return arg
    
    def to_dict(self):
        return self.__dict__
    

REQUIRED_DATA_CONFIG = [
    "name",
    "type",
    "url",
    "labels",
    "stats",
    "item_col",
    "context_features",
    "item_features",
    "train_period/start_date",
    "train_period/end_date",
    "test_period/start_date",
    "test_period/end_date",
]

DEFAULT_CONFIG = {
    "file_format": "auto",
    "date_format": "%Y-%m-%d",
    "user_sequential_info": None,
    "post_process": None,
    "filter_settings": None,
    "item_info": None,
}



@dataclass
class DataArguments(AbsEmbedderDataArguments):
    # Required fields without default values
    name: str=None
    type: str=None
    url: str=None
    file_partition: Dict[str, str]= field(default=None, metadata={"required_keys": ["type", "format"]})
    labels: str = field(default=None, metadata={"nargs": "+"})
    stats: Statistics=None
    item_col: str=None
    context_features: str = field(default=None, metadata={"nargs": "+"})
    item_features: str = field(default=None, metadata={"nargs": "+"})
    item_batch_size: int = 2048 # only used for retriever training
    files: str = field(default=None, metadata={"nargs": "+"})

    # Optional fields with default values
    train_period: Dict[str, datetime] = field(default=None, metadata={"required_keys": ["start_date", "end_date"]})
    test_period: Dict[str, datetime] = field(default=None, metadata={"required_keys": ["start_date", "end_date"]})
    user_sequential_info: Optional[Dict[str, Any]] = field(default=None)
    post_process: Optional[Dict[str, Any]] = None
    filter_settings: Optional[Dict[str, Any]] = field(default=None, metadata={"required_keys": ["by", "filter_conditions"]})
    item_info: Optional[Dict[str, Any]] = field(default=None, metadata={"required_keys": ["url", "key", "columns", "use_cols"]})
    seq_features: str = field(default=None, metadata={"nargs": "+"})

    item_pop_dict_path: str = field(default=None)


    def __post_init__(self):
        # Validate required keys in dictionaries after initialization
        for attr_name, required_keys in [
            ("train_period", ["start_date", "end_date"]),
            ("test_period", ["start_date", "end_date"])
        ]:
            attr_value = getattr(self, attr_name)
            missing_keys = [key for key in required_keys if key not in attr_value]
            if missing_keys:
                raise ValueError(f"Missing required keys in {attr_name}: {missing_keys}")

        # Remove whitespace from feature names and update seq_features if user_sequential_info is provided
        self.context_features = [feat.strip() for feat in self.context_features]
        self.item_features = [feat.strip() for feat in self.item_features]
        if self.user_sequential_info and 'use_cols' in self.user_sequential_info:
            self.seq_features = [feat.strip() for feat in self.user_sequential_info['use_cols']]
     
        
    def get_seq_features(self) -> Dict[str, List[str]]:
        """Get all sequence features used by this dataset.

        Return:
            Dict[str: List[str]]: a dictionary mapping sequence feature names to their columns
        """
        seq_features = {}
        for seq_config in self.user_sequential_info:
            seq_name = seq_config["name"]
            seq_features[seq_name] = seq_config["use_cols"]
        return seq_features
        
        
    def to_attr(self) -> DataAttr4Model:
        """ Get the dataset attributes for model

        Return:
            `DataAttr4Model`: dataset attributes for model. See `Nexus.modules.arguments.DataAttr4Model`.
        """
        seq_feats = self.get_seq_features()
        seq_feats_list = list(seq_feats.keys())
        seq_lens = {info["name"]: info["length"] for info in self.user_sequential_info}
        attr = DataAttr4Model(
            fiid=self.item_col,
            flabels=self.labels,
            features=self.context_features + self.item_features + seq_feats_list,
            context_features=self.context_features,
            item_features=self.item_features,
            seq_features=seq_feats,
            seq_lengths=seq_lens,
            num_items=getattr(self.stats, self.item_col),
            stats=self.stats
        )
        return attr
