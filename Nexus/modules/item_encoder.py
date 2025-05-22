
import torch
from collections import OrderedDict
from . import MultiFeatEmbedding, MLPModule

__all__ = [
    "MLPItemEncoder",
]


# class MLPItemEncoder(torch.nn.Module):
#     def __init__(self, data_config, model_config):
#         super(MLPItemEncoder, self).__init__()
        
#         item_emb = MultiFeatEmbedding(
#             features=data_config.item_features,
#             stats=data_config.stats,
#             embedding_dim=model_config.embedding_dim,
#             concat_embeddings=True
#         )
        
#         mlp = MLPModule(
#             mlp_layers=[item_emb.total_embedding_dim] + model_config.mlp_layers,
#             activation_func=model_config.activation,
#             dropout=model_config.dropout,
#             bias=True,
#             batch_norm=model_config.batch_norm,
#             last_activation=False,
#             last_bn=False
#         )
        
#         self.encoder_mlp_sequence = torch.nn.Sequential(OrderedDict([
#             ("item_embedding", item_emb),
#             ("mlp", mlp)
#         ]))

#     def forward(self, x):
#         return self.encoder_mlp_sequence(x)

class MLPItemEncoder(torch.nn.Module):
    def __init__(self, data_config, model_config):
        super(MLPItemEncoder, self).__init__()
        
        item_emb = MultiFeatEmbedding(
            features=data_config.item_features,
            stats=data_config.stats,
            embedding_dim=model_config.embedding_dim,
            concat_embeddings=True,
            combine_embeddings = model_config.combined_embeddings,
        )
        
        mlp = MLPModule(
            mlp_layers=[item_emb.total_embedding_dim] + model_config.mlp_layers,
            activation_func=model_config.activation,
            dropout=model_config.dropout,
            bias=True,
            batch_norm=model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        
        self.encoder_mlp = torch.nn.Sequential(OrderedDict([
            ("item_emb", item_emb),
            ("mlp", mlp)
        ]))

    def forward(self, x):
        return self.encoder_mlp(x)