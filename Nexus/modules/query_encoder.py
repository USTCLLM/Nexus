import torch
from typing import Dict, Optional, OrderedDict, Union
from .aggregate import AverageAggregator, SelfAttentiveAggregator
from .layer import MLPModule
from .embedding import MultiFeatEmbedding


__all__ = ["BaseQueryEncoderWithSeq", "SASRecEncoder", "MLPQueryEncoder"]

def get_seq_data(d: dict, seq_name: Optional[str]):
    """ Get sequence data from a batch.

    Args:
        d (Dict[str: Any]): A dictionary containing the batch of data.
        seq_names (Optional[str]): The names of the sequence to extract. If None, use the default sequence name 'seq'.

    Returns:
       Dict: A dictionary containing the sequence data. If no sequence data, return an empty dictionary.
    
    """
    if seq_name is not None:
        return d[seq_name]
    if "seq" in d:
        return d['seq']
    else:
        return {}
    

__all__ = ["BaseQueryEncoderWithSeq", "SASRecEncoder"]

class BaseQueryEncoderWithSeq(torch.nn.Module):
    def __init__(self, context_embedding, item_encoder, max_seq_lengths, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.item_encoder = item_encoder
        self.context_embedding = context_embedding
        self.max_seq_lengths: dict = max_seq_lengths
        self.seq_names = list(max_seq_lengths.keys())

        self.seq_aggragation = self.get_seq_aggregator()

    
    def get_seq_aggregator(self):
        return AverageAggregator(dim=1)


    def forward(self, batch):
        seq_embs = []
        for seq_name in self.seq_names:
            seq_data = get_seq_data(batch, seq_name)
            seq_emb = self.item_encoder(seq_data)   # BxLxD1
            seq_emb = self.seq_aggragation(seq_emb) # BxD
            seq_embs.append(seq_emb)               # [BxD]
        context_emb = self.context_embedding(batch) # BxD2
        cat_emb = torch.cat(seq_embs + [context_emb], dim=-1)
        return cat_emb
    

    def extra_repr(self):
        return super().extra_repr()
        

    
class SASRecEncoder(BaseQueryEncoderWithSeq):
    def __init__(
            self,
            context_embedding: torch.nn.Embedding,
            item_encoder: torch.nn.Module,
            max_seq_lengths: Dict[str, int],
            embedding_dim: int,
            n_layers: int=1,
            n_heads: int=4,
            hidden_size: int=512,
            dropout: float=0.3,
            activation: Union[str, torch.nn.Module]='relu',
            *args,
            **kwargs
        ):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        super().__init__(context_embedding, item_encoder, max_seq_lengths, *args, **kwargs)


    def get_seq_aggregator(self):
        return torch.nn.ModuleDict({
            seq_name: SelfAttentiveAggregator(
                input_dim=self.embedding_dim,
                max_seq_len=max_seq_len,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                activation=self.activation
            ) for seq_name, max_seq_len in self.max_seq_lengths.items()
        })
        

    def forward(self, batch):
        seq_embs = []
        for seq_name in self.seq_names:
            seq_data: Dict[str, torch.Tensor] = get_seq_data(batch, seq_name)
            seq_emb = self.item_encoder(seq_data)   # BxLxD
            padding_mask = seq_data[list(seq_data.keys())[0]] == 0
            seq_emb = self.seq_aggragation[seq_name](seq_emb, padding_mask) # BxD
            seq_embs.append(seq_emb)               # [BxD]
        context_emb = self.context_embedding(batch) # BxD2
        cat_emb = torch.cat(seq_embs + [context_emb], dim=-1)   # BxD_total
        return cat_emb
    
        
        
# class MLPQueryEncoder(torch.nn.Module):
#     def __init__(self, data_config, model_config, item_encoder):
#         super(MLPQueryEncoder, self).__init__()
        
#         context_emb = MultiFeatEmbedding(
#             features=data_config.context_features,
#             stats=data_config.stats,
#             embedding_dim=model_config.embedding_dim
#         )
        
#         base_encoder = BaseQueryEncoderWithSeq(
#             context_embedding=context_emb,
#             item_encoder=item_encoder,
#             max_seq_lengths = data_config.seq_lengths,
#         )
        
#         output_dim = model_config.mlp_layers[-1] + context_emb.total_embedding_dim
        
#         mlp = MLPModule(
#             mlp_layers=[output_dim] + model_config.mlp_layers,
#             activation_func=model_config.activation,
#             dropout=model_config.dropout,
#             bias=True,
#             batch_norm=model_config.batch_norm,
#             last_activation=False,
#             last_bn=False
#         )
        
#         self.encoder_mlp_sequence = torch.nn.Sequential(OrderedDict([
#             ("encoder", base_encoder),
#             ("mlp", mlp)
#         ]))

#     def forward(self, x):
#         return self.encoder_mlp_sequence(x)
    
class MLPQueryEncoder(torch.nn.Module):
    def __init__(self, data_config, model_config):
        super(MLPQueryEncoder, self).__init__()
        
        context_emb = MultiFeatEmbedding(
            features=data_config.context_features,
            stats=data_config.stats,
            embedding_dim=model_config.embedding_dim,
            concat_embeddings=True,
            combine_embeddings = model_config.combined_embeddings,
        )
        
        mlp = MLPModule(
            mlp_layers=[context_emb.total_embedding_dim] + model_config.mlp_layers,
            activation_func=model_config.activation,
            dropout=model_config.dropout,
            bias=True,
            batch_norm=model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        
        self.encoder_mlp = torch.nn.Sequential(OrderedDict([
            ("context_emb", context_emb),
            ("mlp", mlp)
        ]))

    def forward(self, x):
        return self.encoder_mlp(x)