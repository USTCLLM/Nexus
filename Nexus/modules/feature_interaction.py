import torch
import torch.nn as nn

from copy import deepcopy


__all__ = ["FactorizationMachine", "CrossNetwork", "MultiExperts","MultiSENet"]

class FactorizationMachine(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, field_dim].
        """
        square_of_sum = torch.sum(inputs, dim=1) ** 2
        sum_of_square = torch.sum(inputs ** 2, dim=1)
        ix = 0.5 * (square_of_sum - sum_of_square)
        return torch.sum(ix, dim=1, keepdim=True)   # [batch_size, 1]



class CrossNetwork(torch.nn.Module):
    """ CrossNetwork proposed in "DCN V2: Improved Deep & Cross Network and Practical Lessons
    for Web-scale Learning to Rank Systems".
    """
    def __init__(self, input_dim: int, n_layers: int=4) -> None:
        super().__init__()
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim, bias=True) for _ in range(n_layers)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, field_dim].
        """
        x0 = inputs
        xl = inputs
        for layer in self.linears:
            xl = x0 * layer(xl) + xl
        return xl
    


class MultiExperts(torch.nn.Module):
    def __init__(self, n_experts: int, export_module: torch.nn.Module) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.experts = torch.nn.ModuleList([deepcopy(export_module) for _ in range(n_experts)])


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(inputs))
        return torch.stack(expert_outputs, dim=1)   # [batch_size, n_experts, hidden_dim]


    def extra_repr(self):
        return f"n_experts={self.n_experts}"

class MultiSENet(nn.Module):
    def __init__(self, n_tasks: int, senet_module: torch.nn.Module) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.senets = torch.nn.ModuleList([deepcopy(senet_module) for _ in range(n_tasks)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        senet_outputs = []
        for senet in self.senets:
            senet_outputs.append(senet(inputs))
        return torch.cat(senet_outputs, dim=-1)   # [batch_size, n_task*hidden_dim]