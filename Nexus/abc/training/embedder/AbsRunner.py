import os
import logging
from pathlib import Path
from typing import Tuple, Union
from abc import ABC, abstractmethod
from transformers import set_seed, PreTrainedTokenizer


from .AbsArguments import (
    AbsEmbedderModelArguments,
    AbsEmbedderDataArguments,
    AbsEmbedderTrainingArguments
)
from .AbsTrainer import AbsEmbedderTrainer
from .AbsModeling import AbsEmbedderModel
from .AbsDataset import (
    AbsEmbedderTrainDataset, AbsEmbedderCollator)
from Nexus.abc.training.runner import AbsRunner

logger = logging.getLogger(__name__)


class AbsEmbedderRunner(AbsRunner):
    """Abstract class to run embedding model fine-tuning.

    Args:
        model_args (AbsEmbedderModelArguments): Model arguments
        data_args (AbsEmbedderDataArguments): Data arguments.
        training_args (AbsEmbedderTrainingArguments): Training arguments.
    """
    def __init__(
        self,
        model_args: AbsEmbedderModelArguments,
        data_args: AbsEmbedderDataArguments,
        training_args: AbsEmbedderTrainingArguments
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    @abstractmethod
    def load_model(self, *args, **kwargs) -> AbsEmbedderModel:
        pass
        
    @abstractmethod
    def load_trainer(self) -> AbsEmbedderTrainer:
        """Abstract method to load the trainer.

        Returns:
            AbsEmbedderTrainer: The loaded trainer instance.
        """
        pass

    @abstractmethod
    def load_dataset(self, *args, **kwargs) -> AbsEmbedderTrainDataset:
        pass
    
    @abstractmethod
    def load_data_collator(self) -> AbsEmbedderCollator:
        """Loads the appropriate data collator.

        Returns:
            AbsEmbedderCollator: Loaded data collator.
        """
        pass

    def run(self):
        """
        Executes the training process.
        """
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

        # Training
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        print('Training completed.')
        self.trainer.save_model()
        print('Training completed. save done')


