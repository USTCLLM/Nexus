import torch
from dataclasses import dataclass, field

from Nexus.training.reranker.recommendation.runner import RankerRunner
from Nexus.training.reranker.recommendation.arguments import ModelArguments
from Nexus.training.reranker.recommendation.modeling import BaseRanker
from Nexus.modules import LambdaModule, MLPModule, HStackModule, CrossNetwork
from Nexus.modules import (
    MLPModule, LambdaModule, MultiFeatEmbedding,
    DeepInterestAggregator, CrossNetwork, HStackModule
    )


@dataclass
class DINModelArguments(ModelArguments):
    embedding_dim: int
    combined_embeddings: bool = False
    cross_net_layers: int = 5
    din_hidden_size: int = 36
    din_feature_inter_strategy: str = "mlp"
    deep_cross_combination: str = "parallel"
    mlp_layers: int = field(default=None, metadata={"nargs": "+"})
    activation: str = "relu"
    dropout: float = 0.1
    batch_norm: bool = False


class DINRanker(BaseRanker):
    def get_embedding_layer(self):
        emb = MultiFeatEmbedding(
            features=list(self.data_config.stats.keys()),
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
        if self.model_config.din_feature_inter_strategy == "mlp":
            layer = MLPModule(
                mlp_layers= [self.num_feat * self.model_config.embedding_dim] + self.model_config.mlp_layers,
                activation_func=self.model_config.activation,
                dropout=self.model_config.dropout,
                bias=True,
                batch_norm=self.model_config.batch_norm,
                last_activation=False,
                last_bn=False
            )
        elif self.model_config.din_feature_inter_strategy == "dcn":
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
                last_bn=False
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
        if self.model_config.din_feature_inter_strategy == "mlp":
            input_dim = self.model_config.mlp_layers[-1]
        elif self.model_config.din_feature_inter_strategy == "dcn":
            if self.model_config.deep_cross_combination == "stacked":
                input_dim = self.model_config.mlp_layers[-1]
            else:
                input_dim = self.num_feat * self.model_config.embedding_dim + self.model_config.mlp_layers[-1]
        pred_mlp = MLPModule(
            mlp_layers=[input_dim] + self.model_config.prediction_layers + [1],
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        return pred_mlp


def main():
    data_config_path = "./examples/recommendation/config/data/recflow_ranker.json"
    train_config_path = "./examples/recommendation/models/config/din_train.json"
    model_config_path = "./examples/recommendation/models/config/din_model.json"
    
    runner = RankerRunner(
        model_config_or_path=DINModelArguments.from_json(model_config_path),
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=DINRanker,
    )
    runner.run()


if __name__ == "__main__":
    main()