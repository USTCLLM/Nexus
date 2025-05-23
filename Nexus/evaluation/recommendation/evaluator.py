from collections import defaultdict
from accelerate import Accelerator
import time

import json
from loguru import logger
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Union, Tuple

from Nexus.training.embedder.recommendation.modeling import BaseRetriever
from Nexus.training.reranker.recommendation.modeling import BaseRanker

from Nexus.modules.metrics import get_eval_metrics
from Nexus.abc.evaluation import AbsEvaluator
from .dataset import RecommenderEvalDataLoader
import torch
from torch.utils.data import DataLoader, Dataset
from Nexus.modules.arguments import log_dict
from .arguments import RecommenderEvalArgs, RecommenderEvalModelArgs

from torchrec.distributed import DistributedModelParallel
from dynamic_embedding.wrappers import wrap_dataloader, wrap_dataset


class RecommenderAbsEvaluator(AbsEvaluator):
    """
    Base class of Evaluator.
    
    Args:
        eval_name (str): The experiment name of current evaluation.
        data_loader (AbsEvalDataLoader): The data_loader to deal with data.
        overwrite (bool): If true, will overwrite the existing results.
    """
    def __init__(
        self,
        retriever_data_loader: DataLoader,
        ranker_data_loader: DataLoader,
        item_loader: DataLoader, 
        config: RecommenderEvalArgs,
        model_config: RecommenderEvalModelArgs,
        *args,
        **kwargs,
    ):
        self.retriever_eval_loader = retriever_data_loader
        self.ranker_eval_loader = ranker_data_loader
        self.item_loader = item_loader
        self.item_vectors = None
        self.item_ids = None
        self.config = config
        self.model_config = model_config
        self.accelerator = Accelerator()
    
    def __call__(
        self,
        retriever: Optional[BaseRetriever] = None,
        ranker: Optional[BaseRanker] = None,
        *args,
        **kwargs,
    ):
        if retriever is not None:
            logger.info(f"Retriever evaluation begins.")
            metrics = self.evaluate(retriever)
            logger.info(f"Retriever evaluation done.")
            log_dict(logger, metrics)
        
        if ranker is not None:
            logger.info(f"Ranker evaluation begins.")
            metrics = self.evaluate(ranker)
            logger.info(f"Ranker evaluation done.")
            log_dict(logger, metrics)
        
    
    @torch.no_grad()
    def evaluate(self, model:Union[BaseRetriever, BaseRanker], *args, **kwargs) -> Dict:
        
        model.eval()
        model_type = model.model_type
        model = self.accelerator.prepare(model)
        if model_type == "retriever":
            if model.model_type == 'retriever':
                item_vector_path = os.path.join(self.model_config.retriever_ckpt_path, 'item_vectors.pt')
                if os.path.exists(item_vector_path):
                    logger.info(f"Loading item vectors from {item_vector_path} ...")
                    item_vectors_dict = torch.load(item_vector_path)
                    self.item_vectors = item_vectors_dict['item_vectors'].to(self.accelerator.device)
                    self.item_ids = item_vectors_dict['item_ids'].to('cpu')
                else:
                    logger.info(f"Updating item vectors...")
                    self.item_vectors, self.item_ids = self.update_item_vectors(model)
                    logger.info(f"Updating item vectors done...")
        eval_outputs = []
        eval_total_bs = 0
        eval_loader = self.retriever_eval_loader if model_type == "retriever" else self.ranker_eval_loader
        for eval_step, eval_batch in enumerate(self.accelerator.prepare(eval_loader)):
            start_time = time.time()

            logger.info(f"Evaluation step {eval_step + 1} begins..")
            eval_batch_size = eval_batch[list(eval_batch.keys())[0]].shape[0]
            metrics = self._eval_batch(model, eval_batch, *args, **kwargs)
            eval_outputs.append((metrics, eval_batch_size))
            eval_total_bs += eval_batch_size
            end_time = time.time()
            elapsed_time = end_time - start_time

            logger.info(f"batch cost time: {elapsed_time} sec")
            logger.info(f"Evaluation step {eval_step + 1} done.")
        model = self.accelerator.unwrap_model(model)
        metrics = self.eval_epoch_end(model, eval_outputs)
        self._total_eval_samples = eval_total_bs
        return metrics
    
    def _eval_batch(self, model:Union[BaseRetriever, BaseRanker], batch, *args, **kwargs) -> Dict:
        """ Evaluate the model on a batch, return metrics.

        Args:
            batch (Dict): The input batch.

        Returns:
            Dict: The metrics.
        """
        with torch.no_grad():
            model.eval()
            k = max(self.config.cutoffs) if self.config.cutoffs is not None else None
            model = self.accelerator.unwrap_model(model)
            if model.model_type == 'retriever':
                outputs = model.eval_step(batch, k=k, item_vectors=self.item_vectors, *args, **kwargs)
            else:
                outputs = model.eval_step(batch, k=k, *args, **kwargs)
            outputs = self.accelerator.gather_for_metrics(outputs)
            metrics: dict = self.compute_metrics(model, outputs)
            return metrics
    
    @torch.no_grad()
    def update_item_vectors(self, model:Union[BaseRetriever, BaseRanker]):
        logger.info(f'Update item vectors...')
        model.eval()
        all_item_vectors, all_item_ids = [], []
        model_unwrapped = self.accelerator.unwrap_model(model)
        for item_batch in self.accelerator.prepare(model_unwrapped.item_loader):
            item_vector = model_unwrapped.item_encoder(item_batch)
            all_item_vectors.append(item_vector)
            all_item_ids.append(item_batch[model_unwrapped.fiid])
        all_item_vectors = self.accelerator.gather_for_metrics(all_item_vectors)
        all_item_ids = self.accelerator.gather_for_metrics(all_item_ids)
        all_item_vectors = torch.cat(all_item_vectors, dim=0)
        all_item_ids = torch.cat(all_item_ids, dim=0).cpu()
        return all_item_vectors, all_item_ids
    
    def load_config(self, config: Union[Dict, str]) -> RecommenderEvalArgs:
        if config is None:
            return RecommenderEvalArgs()
        if isinstance(config, RecommenderEvalArgs):
            return config
        if isinstance(config, dict):
            config_dict = config
        elif isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Config should be either a dictionary or a path to a JSON file, got {type(config)} instead.")
        return RecommenderEvalArgs.from_dict(config_dict)
    
    @torch.no_grad()
    def compute_metrics(self, model:Union[BaseRetriever, BaseRanker], output: Tuple):
        """ Compute the metrics given the output of the model.

        Args:
            output (Tuple): The output of the model.

        Returns:
            Dict: The computed metrics.
        """
        metrics: list = get_eval_metrics(self.config.metrics, model.model_type)
        cutoffs = self.config.cutoffs
        output_dict = {}
        if model.model_type == "retriever":
            for metric, func in metrics:
                for cutoff in cutoffs:
                    output_dict[f"{metric}@{cutoff}"] = func(*output, cutoff)
        else:
            output_dict = (output[0].cpu(), output[1].cpu())    # (pred, target)
        return output_dict
    
    @torch.no_grad()
    def eval_epoch_end(self, model:Union[BaseRetriever, BaseRanker], outputs: List[Tuple]) -> Dict:
        """ Aggregate the metrics from the evaluation batch.

        Args:
            outputs (List): The output of the evaluation batch. It is a list of tuples, 
                where the first element is the metrics (Dict) and the second element is the batch size.

        Returns:
            Dict: The aggregated metrics.
        """
        if model.model_type == "retriever":
            metric_list, bs = zip(*outputs)
            bs = torch.tensor(bs)
            out = defaultdict(list)
            for o in metric_list:
                for k, v in o.items():
                    out[k].append(v)
            for k, v in out.items():
                metric = torch.tensor(v)
                out[k] = ((metric * bs).sum() / bs.sum()).item()
            return out
        else:
            # ranker: AUC, Logloss
            out = {}
            output, bs = zip(*outputs)
            pred, target = zip(*output)
            pred = torch.cat(pred, dim=0)   # [N] or [N, K]
            target = torch.cat(target, dim=0)   # [N] or [N, K]
            metrics: list = get_eval_metrics(self.config.metrics, model.model_type)
            if pred.dim() == 2 and target.dim() == 2:
                # multi task
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    labels = self.model.module.flabel
                else:
                    labels = self.model.flabel
                for i, task in enumerate(labels):
                    for metric, func in metrics:
                        _m = func(pred[:, i], target[:, i])
                        out[f"{metric}/{task}"] = _m.item() if isinstance(_m, torch.Tensor) else _m
                # overall metrics
                for metric, func in metrics:
                    avg_metric = sum([out[f"{metric}/{label}"] for label in labels]) / len(labels)
                    out[metric] = avg_metric
            else:   # single task
                for metric, func in metrics:
                    _m = func(pred, target)
                    out[metric] = _m if isinstance(_m , torch.Tensor) else _m 
            out = {m: v.item() if (isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)) else v for m, v in out.items()}
            return out
            
    
class TDERecommenderEvaluator(RecommenderAbsEvaluator):
    
    def __init__(
        self,
        retriever_data_loader: DataLoader,
        ranker_data_loader: DataLoader,
        item_loader: DataLoader, 
        config: RecommenderEvalArgs,
        model_config: RecommenderEvalModelArgs,
        retriever: BaseRetriever,
        ranker: BaseRanker,
        *args,
        **kwargs,
    ):
        self.retriever_eval_loader = retriever_data_loader
        self.ranker_eval_loader = ranker_data_loader
        self.item_loader = item_loader
        self.item_vectors = None
        self.item_ids = None
        self.config = config
        self.model_config = model_config
        self.accelerator = Accelerator()
        
        # wrap and prepare the dataloader 
        if self.retriever_eval_loader is not None:
            self.retriever_eval_loader = self.accelerator.prepare(self.retriever_eval_loader)
            self.retriever_eval_loader = wrap_dataloader(self.retriever_eval_loader, 
                                                         retriever, retriever.module.tde_configs_dict)
            
            # item loader has a bug, we don't use it. 
            # If we prepare it, we need to gather, there is no gather. 
            # If we don't prepare it, we need to move it to the device manually, there is no move operation.
        
        if self.ranker_eval_loader is not None:
            self.ranker_eval_loader = self.accelerator.prepare(self.ranker_eval_loader)
            self.ranker_eval_loader = wrap_dataloader(self.ranker_eval_loader,
                                                      ranker, ranker.module.tde_configs_dict)
        

    @torch.no_grad()
    def evaluate(self, model:DistributedModelParallel, *args, **kwargs) -> Dict:
        model.eval()
        if model.module.model_type == "retriever":
            item_vector_path = os.path.join(self.model_config.retriever_ckpt_path, 'item_vectors.pt')
            if os.path.exists(item_vector_path):
                logger.info(f"Loading item vectors from {item_vector_path} ...")
                item_vectors_dict = torch.load(item_vector_path)
                self.item_vectors = item_vectors_dict['item_vectors'].to(self.accelerator.device)
                self.item_ids = item_vectors_dict['item_ids'].to('cpu')
            else:
                logger.info(f"Updating item vectors...")
                self.item_vectors, self.item_ids = self.update_item_vectors(model)
                logger.info(f"Updating item vectors done...")
        eval_outputs = []
        eval_total_bs = 0
        eval_loader = self.retriever_eval_loader if model.module.model_type == "retriever" else self.ranker_eval_loader
        for eval_step, eval_batch in enumerate(eval_loader):
            logger.info(f"Evaluation step {eval_step + 1} begins..")
            eval_batch_size = eval_batch[list(eval_batch.keys())[0]].shape[0]
            metrics = self._eval_batch(model.module, eval_batch, *args, **kwargs)
            eval_outputs.append((metrics, eval_batch_size))
            eval_total_bs += eval_batch_size
            logger.info(f"Evaluation step {eval_step + 1} done.")

        metrics = self.eval_epoch_end(model.module, eval_outputs)
        self._total_eval_samples = eval_total_bs
        return metrics