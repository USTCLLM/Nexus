from transformers import HfArgumentParser

from . import (
    DecoderOnlyEmbedderModelArguments,
    DecoderOnlyEmbedderRunner,
)
from .arguments import DecoderOnlyEmbedderDataArguments, DecoderOnlyEmbedderTrainingArguments


def main():
    data_config_path=''
    train_config_path=''
    model_config_path=''
    
    model_args = DecoderOnlyEmbedderModelArguments.from_json(model_config_path)
    data_args = DecoderOnlyEmbedderDataArguments.from_json(data_config_path)
    training_args = DecoderOnlyEmbedderTrainingArguments.from_json(train_config_path)
    runner = DecoderOnlyEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
