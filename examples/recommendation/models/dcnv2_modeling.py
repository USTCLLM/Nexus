import torch
from dataclasses import dataclass, field

from Nexus.training.reranker.recommendation.runner import RankerRunner
from Nexus.training.reranker.recommendation.arguments import ModelArguments
from Nexus.training.reranker.recommendation.modeling import BaseRanker
from Nexus.modules import AverageAggregator, LambdaModule, MLPModule, HStackModule, CrossNetwork
# from rs4industry.model.ranker import BaseRanker
# from rs4industry.model.module import MLPModule, LambdaModule, AverageAggregator, CrossNetwork, HStackModule


@dataclass
class DCNv2ModelArguments(ModelArguments):
    embedding_dim: int
    combined_embeddings: bool = False
    cross_net_layers: int = 5
    deep_cross_combination: str = "parallel"
    mlp_layers: int = field(default=None, metadata={"nargs": "+"})
    activation: str = "relu"
    dropout: float = 0.1
    batch_norm: bool = False


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
        if self.model_config.deep_cross_combination == "stacked":
            input_dim = self.model_config.mlp_layers[-1]
        else:
            input_dim = self.num_feat * self.model_config.embedding_dim + self.model_config.mlp_layers[-1]
        pred_linear = torch.nn.Linear(
            in_features=input_dim,
            out_features=1,
            bias=False
        )
        # BCELoss is not good for autocast in distributed training, reminded by pytorch
        # sigmoid = torch.nn.Sigmoid()
        # return torch.nn.Sequential(pred_mlp, sigmoid)
        return pred_linear
    

def main():
    data_config_path = "./examples/recommendation/config/data/recflow_ranker.json"
    train_config_path = "./examples/recommendation/models/config/dcnv2_train.json"
    model_config_path = "./examples/recommendation/models/config/dcnv2_model.json"
    
    runner = RankerRunner(
        model_config_or_path=DCNv2ModelArguments.from_json(model_config_path),
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=DCNv2Ranker
    )
    runner.run()


if __name__ == "__main__":
    main()