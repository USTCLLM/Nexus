from typing import Callable
import torch
import torch.nn as nn

from . import get_activation

__all__ = [
    "SENet",
    "MLPModule",
    "ActivationUnit",
    "LambdaModule",
    "HStackModule",
    "FC"
]

class SENet(nn.Module):
    def __init__(self, input_dim, reduction, use_bn, droprate_rate):
        super(SENet, self).__init__()
        self.use_bn = use_bn
        
        hidden_dim = input_dim // reduction
        
        self.dp1 = nn.Dropout(droprate_rate)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        
        self.dp2 = nn.Dropout(droprate_rate)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        if self.use_bn:
            self.bn2 = nn.BatchNorm1d(input_dim)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(self.dp1(x))
        if self.use_bn:
            out = self.bn1(out)
        out = self.act1(out)

        out = self.fc2(self.dp2(out))
        if self.use_bn:
            out = self.bn2(out)
        out = self.act2(out)

        return 2 * out * x

class FC(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn, droprate_rate):
        super(FC, self).__init__()
        self.use_bn = use_bn
        
        self.dp = nn.Dropout(droprate_rate)
        self.fc = nn.Linear(input_dim, output_dim)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        out = self.fc(self.dp(x))
        if self.use_bn:
            out = self.bn(out)
        out = self.act(out)
        
        return out

class MLPModule(torch.nn.Module):
    def __init__(
            self, 
            mlp_layers, 
            activation_func='ReLU', 
            dropout=0.0, 
            bias=True, 
            batch_norm=False,
            last_activation=True,
            last_bn=True
        ):
        """
        MLPModule
        Gets a MLP easily and quickly.

        Args:
            mlp_layers(list): the dimensions of every layer in the MLP.
            activation_func(torch.nn.Module,str,None): the activation function in each layer.
            dropout(float): the probability to be set in dropout module. Default: ``0.0``.
            bias(bool): whether to add batch normalization between layers. Default: ``False``.
            last_activation(bool): whether to add activation in the last layer. Default: ``True``.
            last_bn(bool): whether to add batch normalization in the last layer. Default: ``True``.

        Examples:
        >>> MLP = MLPModule([64, 64, 64], 'ReLU', 0.2)
        >>> MLP.model
        Sequential(
            (0): Dropout(p=0.2, inplace=False)
            (1): Linear(in_features=64, out_features=64, bias=True)
            (2): ReLU()
            (3): Dropout(p=0.2, inplace=False)
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): ReLU()
        )
        >>> MLP.add_modules(torch.nn.Linear(64, 10, True), torch.nn.ReLU())
        >>> MLP.model
        Sequential(
            (0): Dropout(p=0.2, inplace=False)
            (1): Linear(in_features=64, out_features=64, bias=True)
            (2): ReLU()
            (3): Dropout(p=0.2, inplace=False)
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): ReLU()
            (6): Linear(in_features=64, out_features=10, bias=True)
            (7): ReLU()
        )
        """
        super().__init__()
        self.mlp_layers = mlp_layers
        self.batch_norm = batch_norm
        self.bias = bias
        self.dropout = dropout
        self.activation_func = activation_func
        self.model = []
        last_bn = self.batch_norm and last_bn
        for idx, layer in enumerate((zip(self.mlp_layers[: -1], self.mlp_layers[1:]))):
            self.model.append(torch.nn.Dropout(dropout))
            self.model.append(torch.nn.Linear(*layer, bias=bias))
            if (idx == len(mlp_layers)-2 and last_bn) or (idx < len(mlp_layers)-2 and batch_norm):
                self.model.append(torch.nn.BatchNorm1d(layer[-1]))
            if ( (idx == len(mlp_layers)-2 and last_activation and activation_func is not None)
                or (idx < len(mlp_layers)-2 and activation_func is not None) ):
                activation = get_activation(activation_func, dim=layer[-1])
                self.model.append(activation)
        self.model = torch.nn.Sequential(*self.model)

    def add_modules(self, *args):
        """
        Adds modules into the MLP model after obtaining the instance.

        Args:
            args(variadic argument): the modules to be added into MLP model.
        """
        for block in args:
            assert isinstance(block, torch.nn.Module)

        for block in args:
            self.model.add_module(str(len(self.model._modules)), block)

    def forward(self, x):
        shape = x.shape
        output = self.model(x.view(-1, shape[-1]))
        return output.view(*shape[:-1], -1)

    # def __repr__(self):
    #     return (f"mlp_layers={self.mlp_layers}, activation_func={self.activation_func},\n"
    #            f"\tdropout={self.dropout},\nbatch_norm={self.batch_norm},\n"
    #            f"\tlast_activation={self.last_activation}\n"
    #            f"\tbias={self.bias}")


class ActivationUnit(torch.nn.Module):
    """ Attention layer to get the relevance of historical items purchased to the target item
    Input shape:
        - tensor of shape ```(batch_size, timestep, embedding_dim)```
    Output shape:
        - tensor of shape ```(batch_size, timestep, 1)```
    """
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        # use linear layer as a proxy for a deep neural network for simplicity
        self.dnn = torch.nn.Linear(4 * input_dim, hidden_size)
        self.dice = get_activation("dice", hidden_size)
        self.dense = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        """
        Args:
            query(torch.Tensor): the query vector with shape (batch_size, ...)
            keys(torch.Tensor): the key vectors with shape (batch_size, seq_len, ...)
        
        Returns:
            attention_score(torch.Tensor): the attention scores with shape (batch_size, seq_len, 1)
        """
        keys = keys.view(*keys.shape[:2], -1)   # (B, L, ...) -> (B, L, D)
        query = query.view(query.size(0), -1)   # (B, ...) -> (B, D)
        sequence_length = keys.size(1) # get sequence length
        query = query.unsqueeze(1).repeat(1, sequence_length, 1) # (B, 1, D) -> (B, L, D)
        attention_input = torch.cat([query, keys, query-keys, query * keys], dim=-1) # (B, L, E * 4)
        attention_output = self.dice(self.dnn(attention_input)) # (B,T, hidden_units[-1])
        # attention score represents the attention score from each keys timestep to the query
        attention_score = self.dense(attention_output) # (B, T, hidden_units[-1]) -> (B, T , 1)
        return attention_score


class LambdaModule(torch.nn.Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    

class HStackModule(torch.nn.Module):
    def __init__(self, modules: list, aggregate_function):
        super().__init__()
        self.layers = torch.nn.ModuleList(modules)
        self.aggregate_function = aggregate_function

    def forward(self, x):
        layer_outputs = []
        for layer in self.layers:
            layer_outputs.append(layer(x))
        return self.aggregate_function(layer_outputs)