import os
from typing import Optional
from dataclasses import dataclass, field

from Nexus.abc.training.embedder import AbsEmbedderModelArguments, AbsEmbedderDataArguments, AbsEmbedderTrainingArguments
from Nexus.abc.training.arguments import AbsModelArguments


# for runner
@dataclass
class TextEmbedderModelArguments(AbsEmbedderModelArguments):
    """
    Abstract class for model arguments.
    """

    config_name: str = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where do you want to store the pre-trained models downloaded from s3."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code"}
    )
    token: str = field(
        default_factory=lambda: os.getenv('HF_TOKEN', None),
        metadata={"help": "The token to use when accessing the model."}
    )


@dataclass
class TextEmbedderDataArguments(AbsEmbedderDataArguments):
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated."
        },
    )

    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set will pad the sequence to be a multiple of the provided value."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query"}
    )
    query_instruction_format: str = field(
        default="{}{}", metadata={"help": "format for query instruction"}
    )

    knowledge_distillation: bool = field(
        default=False,
        metadata={"help": "Use knowledge distillation when `pos_scores: List[float]` and `neg_scores: List[float]` are in features of training data"}
    )

    passage_instruction_for_retrieval: Optional[str] = field(
        default=None, metadata={"help": "instruction for passage"}
    )
    passage_instruction_format: Optional[str] = field(
        default="{}{}", metadata={"help": "format for passage instruction"}
    )

    shuffle_ratio: float = field(
        default=0.0, metadata={"help": "The ratio of shuffling the text"}
    )

    # Parameters for SameDatasetDataArguments
    same_dataset_within_batch: bool = field(
        default=False, metadata={"help": "All samples in the same batch comes from the same dataset."}
    )
    small_threshold: int = field(
        default=0,
        metadata={"help": "The threshold of small dataset. All small dataset in the same directory will be merged into one dataset."}
    )
    drop_threshold: int = field(
        default=0,
        metadata={"help": "The threshold for dropping merged small dataset. If the number of examples in the merged small dataset is less than this threshold, it will be dropped."}
    )

@dataclass
class TextEmbedderTrainingArguments(AbsEmbedderTrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02, metadata={"help": "temperature used for similarity score"})
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method. Available options: cls, mean, last_token. Default: cls", "choices": ['cls', 'mean', 'last_token']})
    normalize_embeddings: bool = field(default=True, metadata={"help": "whether to normalize the embeddings"})
    sub_batch_size: Optional[int] = field(default=None, metadata={"help": "sub batch size for training"})
    kd_loss_type: str = field(default='kl_div', metadata={"help": "the loss type for knowledge distillation. Available options: kl_div, m3_kd_loss. Default: kl_div.", "choices": ['kl_div', 'm3_kd_loss']})


@dataclass
class WrappedTextEmbedderModelArguments(AbsModelArguments):
    """
    class for BiTextEmbedderModel
    """
    negatives_cross_device: bool = field(
        default=False,
        metadata={"help": "Whether to use negatives from other devices."}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for the softmax function."}
    )
    sub_batch_size: int = field(
        default=-1,
        metadata={"help": "Size of sub-batches for processing."}
    )
    kd_loss_type: str = field(
        default="kl_div",
        metadata={"help": "Type of knowledge distillation loss to use."}
    )
    sentence_pooling_method: str = field(
        default="cls",
        metadata={"help": "Method for pooling sentence embeddings."}
    )
    normalize_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to normalize embeddings."}
    )
