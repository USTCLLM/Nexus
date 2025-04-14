from typing import Dict, Union
import torch

__all__ = [
    "MultiFeatEmbedding",
]

class MultiFeatEmbedding(torch.nn.Module):
    def __init__(self, features, stats, embedding_dim, concat_embeddings=True,
                 stack_embeddings=False, combine_embeddings=False, *args, **kwargs):
        """ Embedding layer for multiple features.

        Args:
            features (list): list of feature names
            stats (object): object containing statistics for each feature
            embedding_dim (int): dimension of the embedding vectors
            concat_embeddings (bool): whether to concatenate all embeddings into one tensor or 
                return them separately in a dict. Defaults to True.
            stack_embeddings (bool): whether to stack all embeddings into one tensor along the last dimension or 
                return them separately in a dict. Defaults to False.
            combined_embeddings (bool): whether to combine all embeddings into one table or
                return them separately in a dict. Defaults to False.
                
            .. note::
            `concat_embeddings` and `stack_embeddings` are mutually exclusive. And if both are False, the embeddings are returned in a dict.
        """
        super().__init__(*args, **kwargs)
        self.feat2number = {
            f: getattr(stats, f) for f in features
        }
        self.embedding_dim = embedding_dim
        self.combine_embeddings = combine_embeddings
        if self.combine_embeddings:
            self.feat2offset = {}
            cur_offset = 0 
            for feat, n in self.feat2number.items():
                self.feat2offset[feat] = cur_offset
                cur_offset += n 
            self.combined_table = torch.nn.Embedding(
                num_embeddings=sum(self.feat2number.values()),
                embedding_dim=embedding_dim)
        else:
            self.feat2embedding = torch.nn.ModuleDict({
                feat: torch.nn.Embedding(num_embeddings=n, embedding_dim=embedding_dim, padding_idx=0)
                for feat, n in self.feat2number.items()
            })
        
        self.total_embedding_dim = embedding_dim * len(features)
        # concat_embeddings and stack_embeddings are mutually exclusive
        assert not (concat_embeddings and stack_embeddings), "concat_embeddings and stack_embeddings are mutually exclusive"
        self.concat_embeddings = concat_embeddings
        self.stack_embeddings = stack_embeddings


    def forward(self, batch, strict=True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            batch (dict): a dict of tensors with the same shape as the input to the model
            strict (bool): whether to raise an error if the batch does not contain features that are in the embedding layer.
                To encode sequence features, strict can be set to False. Defaults to True.
        Returns:
            torch.Tensor | Dict[str, torch.Tensor]: either a single tensor with shape [batch_size, total_embedding_dim] 
            or a dictionary with keys being feature names and values being their corresponding embeddings
        """
        outputs = {}
        if self.combine_embeddings:
            field2shape = {} 
            field2batch_offset = {} 
            flattened_batch = []
            cur_batch_offset = 0
            if strict:
                for feat, _ in self.feat2number.items():
                    field2shape[feat] = batch[feat].shape
                    field2batch_offset[feat] = cur_batch_offset
                    flattened_batch.append(batch[feat].flatten() + self.feat2offset[feat]) 
                    cur_batch_offset += torch.prod(torch.tensor(batch[feat].shape)).item()
            else:
                for feat, value in batch.items():
                    if feat in self.feat2number:
                        field2shape[feat] = value.shape
                        field2batch_offset[feat] = cur_batch_offset
                        flattened_batch.append(value.flatten() + self.feat2offset[feat]) # [B] or [B * L]
                        cur_batch_offset += torch.prod(torch.tensor(value.shape)).item()
            flattened_batch = torch.cat(flattened_batch, dim=0) # [F * B * L] or [F * B]
            # get embeddings from combined table in a single call
            flattened_batch_embs = self.combined_table(flattened_batch) # [F * B, embedding_dim] or [F * B * L, embedding_dim]
            for feat, shape in field2shape.items():
                batch_offset = field2batch_offset[feat]
                outputs[feat] = flattened_batch_embs[batch_offset : batch_offset + torch.prod(torch.tensor(shape)).item()]
                outputs[feat] = outputs[feat].reshape(shape + (self.embedding_dim,)) # [B, L, embedding_dim] or [B, embedding_dim]
        else:
            if strict:
                for feat, emb in self.feat2embedding.items():
                    outputs[feat] = emb(batch[feat])
            else:
                for feat, value in batch.items():
                    if feat in self.feat2embedding:
                        outputs[feat] = self.feat2embedding[feat](value)

        if self.concat_embeddings:
            outputs = torch.cat([outputs[f] for f in outputs], dim=-1)  # [*, num_features * embedding_dim]
        elif self.stack_embeddings:
            outputs = torch.stack([outputs[f] for f in outputs], dim=-2) # [*, num_features, embedding_dim]
        return outputs


class TDEMultiFeatEmbedding(torch.nn.Module):

    def __init__(self, multi_feat_embedding: MultiFeatEmbedding, tde_table_configs: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        from torchrec import EmbeddingCollection, EmbeddingConfig

        self.multi_feat_embedding = multi_feat_embedding
        self.features = self.multi_feat_embedding.feat2number.keys()
        
        for k, table_config in tde_table_configs.items():
            if k in self.multi_feat_embedding.feat2embedding:
                self.add_module(f"table_{k}", EmbeddingCollection(tables=[EmbeddingConfig(**table_config)], device=torch.device("meta")))
        
        for k, table_config in tde_table_configs.items():
            for feat in table_config['feature_names']:
                if feat in self.multi_feat_embedding.feat2embedding:
                    self.multi_feat_embedding.feat2embedding.pop(feat)
        

    def forward(self, batch:dict, strict=True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        from torchrec import JaggedTensor, KeyedJaggedTensor
        
        # raw multi_feat_embedding layer
        outputs = {}
        if strict:
            for feat in self.features:
                if hasattr(self, f'table_{feat}'):
                    if batch[feat].dim() == 1:
                        feat_jt = JaggedTensor.from_dense([batch[feat]])
                    elif batch[feat].dim() >= 2:
                        feat_jt = JaggedTensor.from_dense(list(batch[feat]))
                    else:
                        raise ValueError(f"Invalid dimension of {feat}: {batch[feat].dim()}")
                    outputs[feat] = getattr(self, f'table_{feat}')(
                        KeyedJaggedTensor.from_jt_dict({feat : feat_jt}))[feat].to_padded_dense()
                    if outputs[feat].shape[0] == 1:
                        outputs[feat] = outputs[feat].squeeze(0)
                else:
                    outputs[feat] = self.multi_feat_embedding.feat2embedding[feat](batch[feat])
        else:
            for feat, _ in batch.items():
                if feat in self.multi_feat_embedding.feat2embedding or hasattr(self, f'table_{feat}'):
                    if hasattr(self, f'table_{feat}'):
                        if batch[feat].dim() == 1:
                            feat_jt = JaggedTensor.from_dense([batch[feat]])
                        elif batch[feat].dim() >= 2:
                            feat_jt = JaggedTensor.from_dense(list(batch[feat]))
                        else:
                            raise ValueError(f"Invalid dimension of {feat}: {batch[feat].dim()}")
                        outputs[feat] = getattr(self, f'table_{feat}')(
                            KeyedJaggedTensor.from_jt_dict({feat : feat_jt}))[feat].to_padded_dense()
                        if outputs[feat].shape[0] == 1:
                            outputs[feat] = outputs[feat].squeeze(0)
                    else:
                        outputs[feat] = self.multi_feat_embedding.feat2embedding[feat](batch[feat])
        
        if self.multi_feat_embedding.concat_embeddings:
            outputs = torch.cat([outputs[f] for f in outputs], dim=-1)  # [*, num_features * embedding_dim]
        elif self.multi_feat_embedding.stack_embeddings:
            outputs = torch.stack([outputs[f] for f in outputs], dim=-2) # [*, num_features, embedding_dim]
        return outputs