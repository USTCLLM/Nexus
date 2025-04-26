import logging
from typing import Optional
from pathlib import Path
import os
import logging
from typing import Optional

import torch
from transformers import (
    AutoConfig, AutoTokenizer, PreTrainedTokenizer,
    set_seed
)
from Nexus.abc.training.embedder import (
    AbsEmbedderRunner, AbsEmbedderModel, AbsEmbedderTrainDataset,
    AbsEmbedderTrainer
)


from .arguments import DecoderOnlyEmbedderModelArguments, DecoderOnlyEmbedderDataArguments, DecoderOnlyEmbedderTrainingArguments,WrappedDecoderOnlyEmbedderModelArguments
from .dataset import AbsTextEmbedderTrainDataset, AbsTextEmbedderCollator, AbsEmbedderSameDatasetTrainDataset, AbsEmbedderSameDatasetCollator
from .callback import EmbedderTrainerCallbackForDataRefresh
from .trainer import DecoderOnlyEmbedderTrainer
from .modeling import BiDecoderOnlyEmbedderModel
from .load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)


class DecoderOnlyEmbedderRunner(AbsEmbedderRunner):
    """Runner class for decoder only embedding model.

    Args:
        model_args (DecoderOnlyEmbedderModelArguments): Model arguments instance.
        data_args (AbsEmbedderDataArguments): Data arguments instance.
        training_args (AbsEmbedderTrainingArguments): Trainer arguments.
    """
    def __init__(
        self,
        model_args: DecoderOnlyEmbedderModelArguments,
        data_args: DecoderOnlyEmbedderDataArguments,
        training_args: DecoderOnlyEmbedderTrainingArguments,
        model: Optional[AbsEmbedderModel] = None,
        train_dataset: Optional[AbsEmbedderTrainDataset] = None,
        trainer: Optional[AbsEmbedderTrainer] = None,
        loss_function: Optional[torch.nn.Module] = None,
        score_function: Optional[torch.nn.Module] = None
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.wrapped_model_args = WrappedDecoderOnlyEmbedderModelArguments(
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings
        )

        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

        # Set seed
        set_seed(training_args.seed)

        self.loss_function = loss_function
        self.score_function = score_function
        self.model = model if model is not None else self.load_model()
        self.tokenizer=self.model.tokenizer
        self.train_dataset = train_dataset if train_dataset is not None else self.load_dataset()
        self.data_collator = self.load_data_collator()
        self.trainer = trainer if trainer is not None else self.load_trainer()

    def load_model(self) -> BiDecoderOnlyEmbedderModel:
        """Load tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=False,
            add_eos_token=True,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'

        resize = False
        if self.model_args.additional_special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': self.model_args.additional_special_tokens}
            add_num = tokenizer.add_special_tokens(special_tokens_dict)
            if add_num > 0:
                resize = True
                logger.info(f"Add {add_num} special tokens to the tokenizer. Special tokens: {self.model_args.additional_special_tokens}")
            else:
                logger.warning(f"Special tokens {self.model_args.additional_special_tokens} already exists in the tokenizer.")
        base_model = get_model(self.model_args, self.training_args.output_dir, resize, len(tokenizer))

        num_labels = 1
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        logger.info('Config: %s', config)

        model = BiDecoderOnlyEmbedderModel(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
        return model

    def load_trainer(self) -> DecoderOnlyEmbedderTrainer:
        """Load the trainer.

        Returns:
            DecoderOnlyEmbedderTrainer: Loaded trainer instance.
        """
        trainer = DecoderOnlyEmbedderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
        return trainer

    def load_dataset(self) -> AbsTextEmbedderTrainDataset:
        """Loads the training dataset based on data arguments.

        Returns:
            AbsEmbedderTrainDataset: The loaded dataset instance.
        """
        if self.data_args.same_dataset_within_batch:
            train_dataset = AbsEmbedderSameDatasetTrainDataset(
                args=self.data_args,
                default_batch_size=self.training_args.per_device_train_batch_size,
                seed=self.training_args.seed,
                tokenizer=self.tokenizer,
                process_index=self.training_args.process_index,
                num_processes=self.training_args.world_size
            )
            self.training_args.per_device_train_batch_size = 1
            self.training_args.dataloader_num_workers = 0   # avoid multi-processing
        else:
            train_dataset = AbsTextEmbedderTrainDataset(
                args=self.data_args,
                tokenizer=self.tokenizer
            )
        return train_dataset

    def load_data_collator(self) -> AbsTextEmbedderCollator:
        """Loads the appropriate data collator.

        Returns:
            AbsEmbedderCollator: Loaded data collator.
        """
        if self.data_args.same_dataset_within_batch:
            EmbedCollator = AbsEmbedderSameDatasetCollator
        else:
            EmbedCollator = AbsTextEmbedderCollator

        data_collator = EmbedCollator(
            tokenizer=self.tokenizer,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            sub_batch_size=self.training_args.sub_batch_size,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            padding=True,
            return_tensors="pt"
        )
        return data_collator

    def run(self):
        """
        Run the finetune.
        """
        if not self.model_args.only_merge_lora_model:
            Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

            # Training
            self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
            self.trainer.save_model()

        # save merged model
        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)