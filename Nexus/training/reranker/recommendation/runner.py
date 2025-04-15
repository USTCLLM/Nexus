from typing import Tuple, Union
from Nexus.abc.training.reranker import AbsRerankerRunner
from .arguments import TrainingArguments, ModelArguments, DataArguments, DataAttr4Model
from .modeling import BaseRanker
from .trainer import RankerTrainer
from .dataset import AbsRecommenderRerankerCollator, ConfigProcessor, ShardedDataset, ShardedDatasetPA
from Nexus.modules.optimizer import get_lr_scheduler, get_optimizer
from .callback import StopCallback, LoggerCallback
from transformers import PrinterCallback


class RankerRunner(AbsRerankerRunner):
    """
    Finetune Runner for base embedding models.
    """
    def __init__(
        self,
        model_config_or_path: Union[str, ModelArguments],
        data_config_or_path: Union[str, DataArguments],
        train_config_or_path: Union[str, TrainingArguments],
        model_class: BaseRanker,
        model=None,
        trainer=None,
        *args,
        **kwargs,
    ):        
        self.model_class = model_class
        
        self.data_args = DataArguments.from_json(data_config_or_path) if isinstance(data_config_or_path, str) else data_config_or_path
        self.model_args = ModelArguments.from_json(model_config_or_path) if isinstance(model_config_or_path, str) else model_config_or_path
        self.training_args = TrainingArguments.from_json(train_config_or_path) if isinstance(train_config_or_path, str) else train_config_or_path
        
        self.train_dataset, self.cp_attr = self.load_dataset()
        self.model = model if model is not None else self.load_model()
        self.data_collator = self.load_data_collator()
        self.trainer = trainer if trainer is not None else self.load_trainer()

    def load_dataset(self) -> Tuple[ShardedDataset, DataAttr4Model]:
        config_processor = ConfigProcessor(self.data_args)
        train_config, eval_config = config_processor.split_config()

        train_data = ShardedDatasetPA(train_config, self.training_args.train_batch_size, shuffle=True)
        attr = train_config.to_attr()
        if train_data.item_feat_dataset is not None:
            # when candidate item dataset is given, the number of items is set to the number of items in the dataset
            # instead of the max item id in the dataset
            attr.num_items = len(train_data.item_feat_dataset)
        self.training_args.train_batch_size = 1
        self.training_args.remove_unused_columns = False
        return train_data, attr
    
    def load_model(self) -> BaseRanker:
        model = self.model_class(self.cp_attr, self.model_args)
        return model

    def load_trainer(self) -> RankerTrainer:    
        
        self.optimizer = get_optimizer(
            self.training_args.optim,
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay    
        )
        self.lr_scheduler = get_lr_scheduler()
        # self.training_args.dataloader_num_workers = 0   # avoid multi-processing

        trainer = RankerTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            optimizers=[self.optimizer, self.lr_scheduler]
        )
        trainer.add_callback(StopCallback)
        trainer.add_callback(LoggerCallback)
        trainer.pop_callback(PrinterCallback)
        return trainer

    def load_data_collator(self) -> AbsRecommenderRerankerCollator:
        # collator = AbsRecommenderRerankerCollator()
        collator = lambda x: x[0]
        return collator