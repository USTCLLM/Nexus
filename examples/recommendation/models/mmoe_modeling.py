import torch
from dataclasses import dataclass, field

from Nexus.training.reranker.recommendation.runner import RankerRunner
from Nexus.training.reranker.recommendation.arguments import ModelArguments
from Nexus.training.reranker.recommendation.modeling import BaseRanker
from Nexus.modules import LambdaModule, MLPModule, HStackModule, CrossNetwork
from Nexus.modules import (
    MLPModule, LambdaModule, AverageAggregator,
    DeepInterestAggregator, CrossNetwork, HStackModule, MultiExperts
    )


@dataclass
class MMoEModelArguments(ModelArguments):
    embedding_dim: int
    combined_embeddings: bool = False
    n_experts: int = 3
    deep_cross_combination: str = "parallel"
    mlp_layers: int = field(default=None, metadata={"nargs": "+"})
    gate_layers: int = field(default=None, metadata={"nargs": "+"})
    tower_layers: int = field(default=None, metadata={"nargs": "+"})
    activation: str = "relu"
    dropout: float = 0.1
    batch_norm: bool = False


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
        stacked_tower_layers = HStackModule(
            modules=task_towers,
            aggregate_function=lambda x: torch.cat(x, dim=-1)
        )   # [B, T, D] -> [B, T]
        return stacked_tower_layers
    
    
    
def main():
    data_config_path = "./examples/recommendation/config/data/recflow_ranker.json"
    train_config_path = "./examples/recommendation/models/config/mmoe_train.json"
    model_config_path = "./examples/recommendation/models/config/mmoe_model.json"
    
    runner = RankerRunner(
        model_config_or_path=MMoEModelArguments.from_json(model_config_path),
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=MMoERanker,
    )
    runner.run()


if __name__ == "__main__":
    main()