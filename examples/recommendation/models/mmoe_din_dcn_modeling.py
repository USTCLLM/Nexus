import torch
from dataclasses import dataclass, field

from Nexus.training.reranker.recommendation.runner import RankerRunner
from Nexus.training.reranker.recommendation.arguments import ModelArguments
from Nexus.training.reranker.recommendation.modeling import BaseRanker
from Nexus.modules import LambdaModule, MLPModule, HStackModule, CrossNetwork
from Nexus.modules import (
    MLPModule, LambdaModule, MultiFeatEmbedding,
    DeepInterestAggregator, CrossNetwork, HStackModule, MultiExperts
    )


@dataclass
class MMoEDINDCNv2ModelArguments(ModelArguments):
    embedding_dim: int
    combined_embeddings: bool = False
    n_experts: int = 3
    deep_cross_combination: str = "parallel"
    mlp_layers: int = field(default=None, metadata={"nargs": "+"})
    gate_layers: int = field(default=None, metadata={"nargs": "+"})
    tower_layers: int = field(default=None, metadata={"nargs": "+"})
    din_hidden_size: int = 36
    cross_net_layers: int = 3 
    deep_cross_combination: str = "parallel"
    activation: str = "relu"
    dropout: float = 0.1
    batch_norm: bool = False


class MMoEDINDCNv2Ranker(BaseRanker):

    def set_labels(self):   # set all labels as tasks
        return self.data_config.flabels


    def get_embedding_layer(self):
        emb = MultiFeatEmbedding(
            features=self.data_config.stats.columns,
            stats=self.data_config.stats,
            embedding_dim=self.model_config.embedding_dim,
            concat_embeddings=True,
            stack_embeddings=False,
            combine_embeddings=self.model_config.combined_embeddings
        )
        return emb

    def get_sequence_encoder(self):
        seq_feats = self.data_config.seq_features
        encoder_dict = torch.nn.ModuleDict({
            name: DeepInterestAggregator(
                input_dim=len(feats) * self.model_config.embedding_dim, 
                hidden_size=self.model_config.din_hidden_size,
            )
            for name, feats in seq_feats.items()
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
        # pred_mlp_layers = [self.model_config.mlp_layers[-1]] + self.model_config.tower_layers + [1]
        input_dim = self.model_config.mlp_layers[-1]
        cross_net = CrossNetwork(
            input_dim=input_dim,
            n_layers=self.model_config.cross_net_layers,
        )
        deep_net = MLPModule(
            mlp_layers= [input_dim] + self.model_config.mlp_layers,
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=True,
            last_bn=False
        )
        if self.model_config.deep_cross_combination == "stacked":
            layer = torch.nn.Sequential(cross_net, deep_net)
        else:
            layer = HStackModule(
                modules=[cross_net, deep_net], 
                aggregate_function=lambda x: torch.cat(x, dim=-1)
            )
        
        if self.model_config.deep_cross_combination == "stacked":
            input_dim_2 = input_dim
        else:
            input_dim_2 = 2 * input_dim
        pred_linear = torch.nn.Linear(
            in_features=input_dim_2,
            out_features=1,
            bias=False
        )

        task_towers = [
            torch.nn.Sequential(
                LambdaModule(lambda x: x[:, i]),
                layer,
                pred_linear
            )
            for i in range(len(self.flabel))
        ]
        stacked_tower_layers = HStackModule(
            modules=task_towers,
            aggregate_function=lambda x: torch.cat(x, dim=-1)
        )   # [B, T, D] -> [B, T]
        return stacked_tower_layers
    
    
    
    
def main():
    data_config_path = "./examples/recommendation/config/data/recflow_ranker.json"
    train_config_path = "./examples/recommendation/models/config/mmoe_din_dcn_train.json"
    model_config_path = "./examples/recommendation/models/config/mmoe_din_dcn_model.json"
    
    runner = RankerRunner(
        model_config_or_path=MMoEDINDCNv2ModelArguments.from_json(model_config_path),
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=MMoEDINDCNv2Ranker,
    )
    runner.run()


if __name__ == "__main__":
    main()