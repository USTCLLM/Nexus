from dataclasses import dataclass, field
from typing import List, Union

from Nexus.abc.arguments import AbsArguments


@dataclass
class AbsInferenceArguments(AbsArguments):
    stage: str = field(
        default=None,
        metadata={"help": "Stage of the inference process."}
    )
    use_fp16: bool = field(
        default=True,
        metadata={"help": "use fp16"}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Name or path to the model."}
    )
    onnx_model_path: str= field(
        default=None,
        metadata={
            'help':'Path to onnx model'
        }
    )
    trt_model_path: str= field(
        default=None,
        metadata={
            'help':'Path to trt model'
        }
    )
    output_topk: int = field(
        default=10,
        metadata={"help": "Number of top-k results to output."}
    )
    infer_device: Union[str, int, List[int], List[str]] = field(
        default="cpu",
        metadata={"help": "Device to perform inference.", "nargs": "+"}
    )
    infer_mode: str = field(
        default="normal",
        metadata={"help": "Inference mode: normal, onnx, tensorrt.", "choices": ["normal", "onnx", "tensorrt"]}
    )
    infer_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for inference."}
    )
    max_workspace_size: int = field(
        default=1 << 30,
        metadata={
            'help':'Max workspace size for tesorrt session, default = 1 GB'
        }
    )
    normalize: bool = field(
        default=True,
        metadata={
            'help':'normalize embeddings'
        }
    )
    query_max_length: int = field(
        default=512,
        metadata={
            'help':'query max length'
        }
    )
    passage_max_length: int = field(
        default=512,
        metadata={
            'help':'passage max length'
        }
    )