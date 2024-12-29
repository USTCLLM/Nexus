import os
import json
from loguru import logger
from typing import List, Union, Tuple

from .arguments import RecommenderEvalArgs, RecommenderEvalModelArgs
from .evaluator import RecommenderAbsEvaluator
from UniRetrieval.abc.evaluation import AbsEvalRunner
# from UniRetrieval.training.embedder.recommendation.datasets import 

from UniRetrieval.training.embedder.recommendation.modeling import BaseRetriever
from UniRetrieval.training.reranker.recommendation.modeling import BaseRanker
import torch 
from .datasets import RecommenderEvalDataLoader


class RecommenderEvalRunner(AbsEvalRunner):
    
    """
    Abstract class of evaluation runner.
    
    Args:
        eval_args (AbsEvalArgs): :class:AbsEvalArgs object with the evaluation arguments.
        model_args (AbsEvalModelArgs): :class:AbsEvalModelArgs object with the model arguments.
    """
    def __init__(
        self,
        eval_args: RecommenderEvalArgs,
        model_args: RecommenderEvalModelArgs,
    ):
        self.eval_args = eval_args
        self.model_args = model_args

        logger.info(f"Preparing...")
        self.retriever, self.ranker = self.load_retriever_and_ranker(model_args)
        logger.info(f"Loaded retriever and ranker.")
        self.data_loader = self.load_data_loader()
        logger.info(f"Loaded data.")
        self.evaluator = self.load_evaluator()
        logger.info(f"Loaded evaluator.")
        
    def load_retriever_and_ranker(self, model_args: RecommenderEvalModelArgs) -> Tuple[BaseRetriever, Union[BaseRanker, None]]:
        retriever = BaseRetriever.from_pretrained(model_args.retriever_ckpt_path)
        checkpoint = torch.load(os.path.join(model_args.retriever_ckpt_path, 'model.pt'), map_location=torch.device('cpu'))
        retriever.load_state_dict(checkpoint)
        retriever.eval()
    
        ranker = None
        if model_args.ranker_ckpt_path is not None:
            ranker = BaseRanker.from_pretrained(model_args.ranker_ckpt_path)
            checkpoint = torch.load(os.path.join(model_args.ranker_ckpt_path, 'model.pt'), map_location=torch.device('cpu'))
            ranker.load_state_dict(checkpoint)
            ranker.eval()
            
        return retriever, ranker

    def load_data_loader(self) -> RecommenderEvalDataLoader:
        loader = RecommenderEvalDataLoader(self.eval_args)
        return loader

    def load_evaluator(self) -> RecommenderAbsEvaluator:
        evaluator = RecommenderAbsEvaluator(
            data_loader=self.data_loader.eval_loader,
            item_loader=self.data_loader.item_loader,
            config=self.eval_args,
        )
        return evaluator

    def run(self):
        """
        Run the whole evaluation.
        """
        self.evaluator(
            retriever=self.retriever,
            ranker=self.ranker,
        )