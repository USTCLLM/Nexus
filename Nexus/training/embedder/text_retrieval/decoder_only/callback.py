import logging
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl
)

from .arguments import DecoderOnlyEmbedderModelArguments
from .dataset import AbsEmbedderSameDatasetTrainDataset

logger = logging.getLogger(__name__)


class EmbedderTrainerCallbackForDataRefresh(TrainerCallback):
    """
    Callback class to inspect the state of the training loop and take decision.
    """
    def __init__(self, train_dataset: AbsEmbedderSameDatasetTrainDataset):
        self.train_dataset = train_dataset

    def on_epoch_end(
        self,
        args: DecoderOnlyEmbedderModelArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event called at the end of an epoch.
        """
        self.train_dataset.refresh_epoch()
