from torch.utils.data import DataLoader

from Nexus.training.embedder.recommendation.dataset import ConfigProcessor, ShardedDataset, get_datasets
from Nexus.abc.evaluation import AbsEvalDataLoader
from Nexus.evaluation.recommendation.arguments import RecommenderEvalArgs, RecommenderEvalModelArgs
from Nexus.modules.arguments import DataAttr4Model
from Nexus.training.embedder.recommendation.dataset import AbsRecommenderEmbedderCollator

from Nexus.training.reranker.recommendation.dataset import ShardedDatasetPA as RankerDatasetPA
from Nexus.training.embedder.recommendation.dataset import ShardedDatasetPA as RetrieverDatasetPA

from dynamic_embedding.wrappers import wrap_dataloader, wrap_dataset
    


def get_retriever_datasets(config: str, batch_size: int):
    config_processor = ConfigProcessor(config)
    
    _, eval_config = config_processor.split_config()
    
    test_data = RetrieverDatasetPA(
        eval_config, 
        batch_size,
        shuffle=False
    )
    
    attr = eval_config.to_attr()
    
    attr.num_items = len(test_data.item_feat_dataset)
    
    return test_data

def get_reranker_datasets(config: str, batch_size: int):
    config_processor = ConfigProcessor(config)
    
    _, eval_config = config_processor.split_config()

    test_data = RankerDatasetPA(
        eval_config, 
        batch_size,
        shuffle=False
    )
    
    return test_data


class RecommenderEvalDataLoader(AbsEvalDataLoader, DataLoader):
    def __init__(
        self,
        config: RecommenderEvalArgs,
        model_args: RecommenderEvalModelArgs,
    ):
        self.config = config
        self.eval_dataset = None
        self.collator = lambda x: x[0]
        self.data_attr: DataAttr4Model = None
        self.retriever_eval_loader = None
        self.ranker_eval_loader = None
        self.item_loader = None
        
        if model_args.retriever_ckpt_path is not None:
            retriever_eval_dataset = get_retriever_datasets(config.retriever_data_path, config.eval_batch_size)
            # (self.retriever_train_dataset, self.retriever_eval_dataset), self.retriever_data_attr = get_datasets(config.retriever_data_path)
            self.retriever_eval_loader = DataLoader(
                retriever_eval_dataset, 
                batch_size=config.eval_batch_size,
                collate_fn=self.collator
            )
            self.item_loader = DataLoader(
                retriever_eval_dataset.item_feat_dataset, 
                batch_size=config.retriever_item_batch_size,
                num_workers=32,
            )
        
        if model_args.ranker_ckpt_path is not None:
            ranker_eval_dataset = get_reranker_datasets(config.ranker_data_path, config.eval_batch_size)
            # (self.ranker_train_dataset, self.ranker_eval_dataset), self.ranker_data_attr = get_datasets(config.ranker_data_path)
            self.ranker_eval_loader = DataLoader(
                ranker_eval_dataset, 
                batch_size=config.eval_batch_size,
                collate_fn=self.collator
            )
            