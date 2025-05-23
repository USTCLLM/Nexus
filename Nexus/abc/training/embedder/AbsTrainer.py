import logging
import torch
from typing import Optional
from abc import ABC, abstractmethod
from transformers.trainer import Trainer
from Nexus.abc.training.trainer import AbsTrainer
from Nexus.abc.training.dataset import AbsDataset
logger = logging.getLogger(__name__)


class AbsEmbedderTrainer(AbsTrainer):
    """
    Abstract class for the trainer of embedder.
    """
    @abstractmethod
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        pass
    
    def compute_loss(self, model, inputs, return_outputs=False ,*args, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        
        Args:
            model (AbsEmbedderModel): The model being trained.
            inputs (dict): A dictionary of input tensors to be passed to the model.
            return_outputs (bool, optional): If ``True``, returns both the loss and the model's outputs. Otherwise,
                returns only the loss.
        
        Returns:
            Union[torch.Tensor, tuple(torch.Tensor, EmbedderOutput)]: The computed loss. If ``return_outputs`` is ``True``, 
                also returns the model's outputs in a tuple ``(loss, outputs)``.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
    