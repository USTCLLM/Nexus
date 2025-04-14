import torch
from dataclasses import dataclass, field
from collections import OrderedDict

from Nexus.training.embedder.recommendation.runner import RetrieverRunner
from Nexus.training.embedder.recommendation.modeling import BaseRetriever
from Nexus.training.embedder.recommendation.arguments import ModelArguments
from Nexus.modules import (
    MultiFeatEmbedding,
    SASRecEncoder,
    MLPModule,
    InnerProductScorer,
    UniformSampler,
    BinaryCrossEntropyLoss,
)


@dataclass
class SASRecArguments(ModelArguments):
    embedding_dim: int = 2
    combined_embeddings: bool = False
    num_neg: int = 50
    n_layers: int = 1
    n_heads: int = 2
    hidden_size: int = 64
    mlp_layers: int = field(default=None, metadata={"nargs": "+"})
    activation: str = "relu"
    dropout: float = 0.3
    batch_norm: bool = True


class SASRecRetriever(BaseRetriever):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def get_item_encoder(self):
        item_embedding = MultiFeatEmbedding(
            features=self.data_config.item_features,
            stats=self.data_config.stats,
            embedding_dim=self.model_config.embedding_dim,
            concat_embeddings=True, 
            combine_embeddings=self.model_config.combined_embeddings,
        )
        return item_embedding
    

    def get_query_encoder(self):
        context_emb = MultiFeatEmbedding(
            features=self.data_config.context_features,
            stats=self.data_config.stats,
            embedding_dim=self.model_config.embedding_dim,
            combine_embeddings=self.model_config.combined_embeddings
        )
        encoder = SASRecEncoder(
            context_embedding=context_emb,
            item_encoder=self.item_encoder,
            max_seq_lengths = self.data_config.seq_lengths,
            embedding_dim=self.item_encoder.total_embedding_dim,
            n_layers=self.model_config.n_layers,
            n_heads=self.model_config.n_heads,
            hidden_size=self.model_config.hidden_size,
            dropout=self.model_config.dropout,
            activation=self.model_config.activation
        )
        num_seqs = len(self.data_config.seq_lengths)
        output_dim = self.item_encoder.total_embedding_dim * num_seqs + context_emb.total_embedding_dim
        mlp = MLPModule(
            mlp_layers= [output_dim] + self.model_config.mlp_layers + [self.item_encoder.total_embedding_dim],
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        return torch.nn.Sequential(OrderedDict([
            ("encoder", encoder),
            ("mlp", mlp)
        ]))
    

    def get_score_function(self):
        return InnerProductScorer()
    
    
    def get_loss_function(self):
        return BinaryCrossEntropyLoss()

    
    def get_negative_sampler(self):
        return UniformSampler(num_items=self.data_config.num_items)
    
    
def main():
    data_config_path = "./examples/recommendation/config/data/recflow_retriever.json"
    train_config_path = "./examples/recommendation/models/config/sasrec_train.json"
    model_config_path = "./examples/recommendation/models/config/sasrec_model.json"
    
    runner = RetrieverRunner(
        model_config_or_path=SASRecArguments.from_json(model_config_path),
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=SASRecRetriever,
    )
    runner.run()


if __name__ == "__main__":
    main()