import os
import json
from typing import Dict, Union, Tuple, List, Optional
import torch

from Nexus.modules.loss import BCEWithLogitLoss
from Nexus.abc.training.reranker import AbsRerankerModel, RerankerOutput
from Nexus.training.reranker.recommendation.arguments import DataAttr4Model, ModelArguments
from Nexus.modules import MLPModule, LambdaModule, MultiFeatEmbedding, AverageAggregator
from Nexus.modules.arguments import get_model_cls, get_modules, split_batch
from loguru import logger

from Nexus.modules import AverageAggregator, LambdaModule, MLPModule, HStackModule, CrossNetwork, MultiExperts

class BaseRanker(AbsRerankerModel):
    def __init__(
            self, 
            data_config: DataAttr4Model,
            model_config: Union[Dict, str],
            *args, **kwargs
        ):
        # from BaseModel
        
        self.data_config: DataAttr4Model = data_config
        self.model_config: ModelArguments = self.load_config(model_config)
        self.num_seq_feat = sum([len(seq_feats) for seq_feats in data_config.seq_features.values()])
        self.num_context_feat = len(data_config.context_features)
        self.num_item_feat = len(data_config.item_features)
        self.num_feat = self.num_seq_feat + self.num_context_feat + self.num_item_feat
        self.model_type = "ranker"
        self.num_items: int = self.data_config.num_items
        self.fiid: str = self.data_config.fiid  # item id field
        self.flabel: str = self.set_labels()
        super(BaseRanker, self).__init__(*args, **kwargs)
        
        
    def set_labels(self) -> Union[str, List[str]]:
        # label fields, base ranker only support one label
        # if need multiple labels, use MultiTaskRanker instead
        flabel: str = self.data_config.flabels[0]
        return flabel
    
        
    def init_modules(self):
        super().init_modules()
        self.embedding_layer = self.get_embedding_layer()
        self.sequence_encoder = self.get_sequence_encoder()
        self.feature_interaction_layer = self.get_feature_interaction_layer()
        self.prediction_layer = self.get_prediction_layer()


    def get_embedding_layer(self):
        emb = MultiFeatEmbedding(
            features=self.data_config.stats.columns,
            stats=self.data_config.stats,
            embedding_dim=self.model_config.embedding_dim,
            concat_embeddings=False,
            stack_embeddings=True, 
            combine_embeddings=self.model_config.combine_embeddings,
        )
        return emb
    
    def get_sequence_encoder(self):
        raise NotImplementedError


    def get_feature_interaction_layer(self):
        raise NotImplementedError

    
    def get_prediction_layer(self):
        raise NotImplementedError

    def get_loss_function(self):
        # BCELoss is not good for autocast in distributed training, reminded by pytorch
        return BCEWithLogitLoss(reduction='mean')


    def compute_score(
            self, 
            batch, 
            *args, 
            **kwargs
        ) -> RerankerOutput:
        context_feat, item_feat, seq_feat_dict = split_batch(batch, self.data_config)
        all_embs = []
        context_emb = self.embedding_layer(context_feat, strict=False)  # [B, N2, D]
        item_emb = self.embedding_layer(item_feat, strict=False)    # [B, N3, D]
        if len(seq_feat_dict) > 0:
            for seq_name, seq_feat in seq_feat_dict.items():
                padding_mask = seq_feat[self.fiid] == 0
                seq_emb = self.embedding_layer(seq_feat, strict=False)
                seq_rep = self.sequence_encoder[seq_name](
                    seq=seq_emb,
                    padding_mask=padding_mask,
                    target=item_emb,
                )   # [B, N1, D]
                all_embs.append(seq_rep)
        all_embs += [context_emb, item_emb]
        all_embs_cat = torch.concat(all_embs, dim=1) # [B, N1+N2+N3, D]
        interacted_emb = self.feature_interaction_layer(all_embs_cat)    # [B, **]
        score = self.prediction_layer(interacted_emb)   # [B], sigmoid
        if (len(score.shape) == 2) and (score.size(-1) == 1):
            # score = score.squeeze(-1)   # [B, 1] -> [B]
            score = score.reshape(-1)   # [B, 1] -> [B]
        return RerankerOutput(score, all_embs)
    
    
    def get_score_function(self):
        return self.compute_score
        
    def forward(self, batch, cal_loss=False, *args, **kwargs) -> RerankerOutput:
        if cal_loss:
            return self.compute_loss(batch, *args, **kwargs)
        else:
            output = self.compute_score(batch, *args, **kwargs)
        return output

    def compute_loss(self, batch, *args, **kwargs) -> Dict:
        output = self.forward(batch, *args, **kwargs)
        output_dict = output.to_dict()
        if isinstance(self.flabel, str): # single task
            label = batch[self.flabel].float()
        else:
            label = []
            for flabel in self.flabel:
                label.append(batch[flabel].float())
            label = torch.stack(label, dim=1) # [batch_size, n_task]
        output_dict['label'] = label
        loss = self.loss_function(**output_dict)
        # print("output_dict.scores:",output_dict['scores'])
        # print("label:",output_dict['label'])
        if isinstance(loss, dict):
            return loss
        else:
            return {'loss': loss}
        
    @torch.no_grad()
    def eval_step(self, batch, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.compute_score(batch, *args, **kwargs)
        score = output.scores
        # check if the last layer of the model is a sigmoid function
        # get the last layer of the model
        pred = score
        if isinstance(self.flabel, str):    # single task
            target = batch[self.flabel].float()
        else:   # multi task
            target = []
            for flabel in self.flabel:
                target.append(batch[flabel].float())
            target = torch.stack(target, dim=1) # [batch_size, n_task]
        return pred, target

    @torch.no_grad()
    def predict(self, context_input: Dict, candidates: Dict, output_topk=None, *args, **kwargs):
        """ predict topk candidates for each context
        
        Args:
            context_input (Dict): input context feature
            candidates (Dict): candidate items
            topk (int): topk candidates

        Returns:
            torch.Tensor: topk indices (offset instead of real item id)
        """
        num_candidates = candidates[self.fiid].size(1)
        batch_size = candidates[self.fiid].size(0)
        for k, v in context_input.items():
            # B, * -> BxN, *
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    # replace repeat_interleave
                    v_expanded = v_.unsqueeze(1).expand(-1, num_candidates, *v_.shape[1:])  # expand
                    v[k_] = v_expanded.reshape(-1, *v_.shape[1:])  # flatten
            else:
                # replace repeat_interleave
                v_expanded = v.unsqueeze(1).expand(-1, num_candidates, *v.shape[1:])  # expand
                context_input[k] = v_expanded.reshape(-1, *v.shape[1:])  # flatten
                
        for k, v in candidates.items():
            # B, N, * -> BxN, *
            candidates[k] = v.view(-1, *v.shape[2:])
        context_input.update(candidates)    # {key: BxN, *}
        output = self.compute_score(context_input, *args, **kwargs)
        scores = output.scores.view(batch_size, num_candidates)  # [B, N]
        
        # get topk idx
        topk = min(self.model_config.topk, num_candidates)
        topk_score, topk_idx = torch.topk(scores, topk)
        return topk_idx

    
    def load_config(self, config: Union[Dict, str]) -> ModelArguments:
        if isinstance(config, ModelArguments):
            config = config
        elif isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config = ModelArguments.from_dict(config_dict)
        elif isinstance(config, dict):
            config = ModelArguments.from_dict(config)
        else:
            raise ValueError(f"Config should be either a dictionary or a path to a JSON file, got {type(config)} instead.")
        return config


    @staticmethod
    def from_pretrained(checkpoint_dir: str, 
                        model_class_or_name:Optional[Union[type, str]]=None):
        config_path = os.path.join(checkpoint_dir, "model_config.json")
        with open(config_path, "r", encoding="utf-8") as config_path:
            config_dict = json.load(config_path)
            
        # data config 
        data_config = DataAttr4Model.from_dict(config_dict['data_config'])
        del config_dict['data_config']
        
        # model config 
        model_config = ModelArguments.from_dict(config_dict)
        
        # model class 
        if model_class_or_name is None:
            model_class_or_name = config_dict['model_name']
        if isinstance(model_class_or_name, str):
            model_cls = get_model_cls(config_dict['model_type'], model_class_or_name)
        else:
            model_cls = model_class_or_name
        del config_dict['model_type'], config_dict['model_name']
        
        ckpt_path = os.path.join(checkpoint_dir, "model.pt")
        state_dict = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        model = model_cls(data_config, model_config)
        if "item_vectors" in state_dict:
            model.item_vectors = state_dict["item_vectors"]
            del state_dict['item_vectors']
        model.load_state_dict(state_dict=state_dict, strict=True)
        return model


    def save(self, checkpoint_dir: str, **kwargs):
        self.save_checkpoint(checkpoint_dir)
        self.save_configurations(checkpoint_dir)

    def save_checkpoint(self, checkpoint_dir: str):
        path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(self.state_dict(), path)
        
    def save_configurations(self, checkpoint_dir: str):
        path = os.path.join(checkpoint_dir, "model_config.json")
        config_dict = self.model_config.to_dict()
        config_dict['model_name_or_path'] = checkpoint_dir
        config_dict['model_type'] = self.model_type
        config_dict['model_name'] = self.__class__.__name__
        config_dict['data_config'] = self.data_config.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)


class MLPRanker(BaseRanker):
    def __init__(self, ranker_data_config, ranker_model_config, *args, **kwargs):
        super().__init__(ranker_data_config, ranker_model_config, *args, **kwargs)
        
    def get_sequence_encoder(self):
        encoder_dict = torch.nn.ModuleDict({
            name: AverageAggregator(dim=1)
            for name in self.data_config.seq_features
        })
        return encoder_dict
        
    def get_feature_interaction_layer(self):
        flatten_layer = LambdaModule(lambda x: x.flatten(start_dim=1))  # [B, N, D] -> [B, N*D]
        mlp_layer = MLPModule(
            mlp_layers= [self.num_feat * self.model_config.embedding_dim] + self.model_config.mlp_layers,
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        return torch.nn.Sequential(flatten_layer, mlp_layer)
    

    def get_prediction_layer(self):
        pred_mlp = MLPModule(
            mlp_layers=[self.model_config.mlp_layers[-1]] + self.model_config.prediction_layers + [1],
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        return pred_mlp
    
class MyMLPRanker(MLPRanker):
    def __init__(self, ranker_data_config, ranker_model_config, *args, **kwargs):
        super().__init__(ranker_data_config, ranker_model_config, *args, **kwargs)
        self.topk=6 

    def compute_score(self, batch, *args, **kwargs) -> RerankerOutput:
        context_feat, seq_feat, item_feat = split_batch(batch, self.data_config)
        all_embs = []
        if len(seq_feat) > 0:
            seq_emb = self.embedding_layer(seq_feat, strict=False)
            seq_rep = self.sequence_encoder(seq_emb)   # [B, N1, D]
            all_embs.append(seq_rep)
        context_emb = self.embedding_layer(context_feat, strict=False)  # [B, N2, D]
        item_emb = self.embedding_layer(item_feat, strict=False)    # [B, N3, D]
        all_embs += [context_emb, item_emb]
        all_embs = torch.concat(all_embs, dim=1) # [B, N1+N2+N3, D]
        interacted_emb = self.feature_interaction_layer(all_embs)    # [B, **]
        score = self.prediction_layer(interacted_emb)   # [B], sigmoid
        if len(score.shape) == 2 and score.size(-1) == 1:
            score = score.squeeze(-1)   # [B, 1] -> [B]
        return RerankerOutput(score, [context_emb, item_emb, seq_emb])
    
    def forward(self, batch, cal_loss=False, *args, **kwargs) -> RerankerOutput:
        output = self.compute_score(batch, *args, **kwargs)
        return output
    
    @torch.no_grad()
    def predict(self, context_input: Dict, candidates: Dict, topk: int=6, gpu_mem_save=False, *args, **kwargs):
        """ predict topk candidates for each context
        
        Args:
            context_input (Dict): input context feature
            candidates (Dict): candidate items
            topk (int): topk candidates
            gpu_mem_save (bool): whether to save gpu memroy by using loop to process each candidate

        Returns:
            torch.Tensor: topk indices (offset instead of real item id)
        """
        num_candidates = candidates[self.fiid].size(1)
        
        if not gpu_mem_save:
            self.topk=1
            batch_size = candidates[self.fiid].size(0)
            for k, v in context_input.items():
                # B, * -> BxN, *
                if isinstance(v, dict):
                    for k_, v_ in v.items():
                        # 替代 repeat_interleave
                        v_expanded = v_.unsqueeze(1).expand(-1, num_candidates, *v_.shape[1:])  # 扩展
                        v[k_] = v_expanded.reshape(-1, *v_.shape[1:])  # 展平
                else:
                    # 替代 repeat_interleave
                    v_expanded = v.unsqueeze(1).expand(-1, num_candidates, *v.shape[1:])  # 扩展
                    context_input[k] = v_expanded.reshape(-1, *v.shape[1:])  # 展平
                    
            for k, v in candidates.items():
                # B, N, * -> BxN, *
                candidates[k] = v.view(-1, *v.shape[2:])
            context_input.update(candidates)    # {key: BxN, *}
            output = self.compute_score(context_input, *args, **kwargs)
            scores = output.score.view(batch_size, num_candidates)
            
        else:
            scores = []
            for i in range(num_candidates):
                candidate = {k: v[:, i] for k, v in candidates.items()}
                new_batch = dict(**context_input)
                new_batch.update(candidate)
                output = self.compute_score(new_batch, *args, **kwargs)
                scores.append(output.score)
            scores = torch.stack(scores, dim=-1)  
        # get topk idx
        
        self.topk = min(self.topk, num_candidates)
        topk_score, topk_idx = torch.topk(scores, self.topk)
        return topk_idx
    

class DCNv2Ranker(BaseRanker):
    def get_sequence_encoder(self):
        encoder_dict = torch.nn.ModuleDict({
            name: AverageAggregator(dim=1)
            for name in self.data_config.seq_features
        })
        return encoder_dict
        
    
    def get_feature_interaction_layer(self):
        flatten_layer = LambdaModule(lambda x: x.flatten(start_dim=1))  # [B, N, D] -> [B, N*D]
        cross_net = CrossNetwork(
            input_dim=self.num_feat * self.model_config.embedding_dim,
            n_layers=self.model_config.cross_net_layers,
        )
        deep_net = MLPModule(
            mlp_layers= [self.num_feat * self.model_config.embedding_dim] + self.model_config.mlp_layers,
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=True,
            last_bn=True
        )
        if self.model_config.deep_cross_combination == "stacked":
            layer = torch.nn.Sequential(cross_net, deep_net)
        else:
            layer = HStackModule(
                modules=[cross_net, deep_net], 
                aggregate_function=lambda x: torch.cat(x, dim=-1)
            )
        return torch.nn.Sequential(flatten_layer, layer)
    

    def get_prediction_layer(self):
        if self.model_config.deep_cross_combination == "stacked":
            input_dim = self.model_config.mlp_layers[-1]
        else:
            input_dim = self.num_feat * self.model_config.embedding_dim + self.model_config.mlp_layers[-1]
        pred_linear = torch.nn.Linear(
            in_features=input_dim,
            out_features=1,
            bias=False
        )
        return pred_linear
    
class MMoERanker(BaseRanker):

    def set_labels(self):   # set all labels as tasks
        return self.data_config.flabels

    def get_sequence_encoder(self):
        encoder_dict = torch.nn.ModuleDict({
            name: AverageAggregator(dim=1)
            for name in self.data_config.seq_features
        })
        return encoder_dict
        
    
    def get_feature_interaction_layer(self):
        flatten_layer = LambdaModule(lambda x: x.flatten(start_dim=1))  # [B, N, D] -> [B, N*D]
        expert_mlp = MLPModule(
            mlp_layers=[self.num_feat * self.model_config.embedding_dim] + self.model_config.mlp_layers,
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=True,
            last_bn=False
        )
        shared_multi_experts = MultiExperts(
            n_experts=self.model_config.n_experts,
            export_module=expert_mlp,
        )   # [B, D] -> [B, N, D]

        gate_mlp_layer = [self.num_feat * self.model_config.embedding_dim] + \
            self.model_config.gate_layers + [self.model_config.n_experts]
        multi_task_gates = HStackModule(
            modules=[
                torch.nn.Sequential(
                    MLPModule(
                        mlp_layers=gate_mlp_layer,
                        activation_func=self.model_config.activation,
                        dropout=self.model_config.dropout,
                        bias=True,
                        batch_norm=self.model_config.batch_norm,
                        last_activation=False,
                        last_bn=False
                    ),
                torch.nn.Softmax(dim=-1),
                )
                for _ in range(len(self.flabel))
            ],
            aggregate_function=lambda x: torch.stack(x, dim=1)
        )  # [B, D] -> [B, T, N]
        
        mmoe_expert_gates = HStackModule(
            modules=[multi_task_gates, shared_multi_experts],
            aggregate_function=lambda x: torch.bmm(x[0], x[1]),
        )   # [B, D] -> [B, T, D]

        return torch.nn.Sequential(flatten_layer, mmoe_expert_gates)
    

    def get_prediction_layer(self):
        pred_mlp_layers = [self.model_config.mlp_layers[-1]] + self.model_config.tower_layers + [1]
        task_towers = [
            torch.nn.Sequential(
                LambdaModule(lambda x: x[:, i]),
                MLPModule(
                    mlp_layers=pred_mlp_layers,
                    activation_func=self.model_config.activation,
                    dropout=self.model_config.dropout,
                    bias=True,
                    batch_norm=self.model_config.batch_norm,
                    last_activation=False,
                    last_bn=False
                )
            )
            for i in range(len(self.flabel))
        ]
        print(f"self.flabel:{self.flabel}")
        stacked_tower_layers = HStackModule(
            modules=task_towers,
            aggregate_function=lambda x: torch.cat(x, dim=-1)
        )   # [B, T, D] -> [B, T]
        return stacked_tower_layers