"""Edge-aware Graph Attention Network implemented with plain PyTorch.

The implementation deliberately avoids torch-geometric so that the research
environment can run on the bundled CPU PyTorch installation.  Communication
edges are directed ``source -> target``; attention is normalized over incoming
edges of each target node.  Virtual self-loops are added inside the network
and are not treated as physical radio links.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_graph_inputs(node_features: Tensor, edge_index: Tensor, edge_features: Tensor) -> None:
    if node_features.ndim != 2:
        raise ValueError(f"node_features must have shape [N,F], got {tuple(node_features.shape)}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2,E], got {tuple(edge_index.shape)}")
    if edge_features.ndim != 2 or edge_features.shape[0] != edge_index.shape[1]:
        raise ValueError(
            "edge_features must have shape [E,F_e] with the same E as edge_index; "
            f"got {tuple(edge_features.shape)} and {tuple(edge_index.shape)}"
        )
    if edge_index.numel():
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= node_features.shape[0]:
            raise ValueError("edge_index contains a node outside node_features")


class EdgeGATLayer(nn.Module):
    """One edge-aware, multi-head GAT layer."""

    def __init__(
        self,
        in_node_dim: int,
        out_node_dim: int,
        edge_dim: int,
        *,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if min(in_node_dim, out_node_dim, edge_dim, heads) <= 0:
            raise ValueError("GAT dimensions and heads must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.in_node_dim = int(in_node_dim)
        self.out_node_dim = int(out_node_dim)
        self.edge_dim = int(edge_dim)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.add_self_loops = bool(add_self_loops)
        self.negative_slope = float(negative_slope)

        self.node_projection = nn.Linear(in_node_dim, heads * out_node_dim, bias=False)
        self.edge_projection = nn.Linear(edge_dim, heads * out_node_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, 3 * out_node_dim))
        output_dim = heads * out_node_dim if concat else out_node_dim
        self.residual = nn.Linear(in_node_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        self.last_attention: tuple[Tensor, Tensor, Tensor] | None = None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_projection.weight)
        nn.init.xavier_uniform_(self.edge_projection.weight)
        nn.init.xavier_uniform_(self.residual.weight)
        nn.init.xavier_uniform_(self.attention)
        nn.init.zeros_(self.bias)

    def _add_self_loops(
        self,
        node_count: int,
        edge_index: Tensor,
        edge_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self.add_self_loops:
            return edge_index, edge_features
        device = edge_index.device
        nodes = torch.arange(node_count, dtype=torch.long, device=device)
        loops = torch.stack([nodes, nodes], dim=0)
        loop_features = torch.zeros(
            (node_count, self.edge_dim), dtype=edge_features.dtype, device=edge_features.device
        )
        return (
            torch.cat([edge_index, loops], dim=1),
            torch.cat([edge_features, loop_features], dim=0),
        )

    def _target_softmax(self, logits: Tensor, targets: Tensor, node_count: int) -> Tensor:
        """Softmax logits independently for each target and attention head."""

        result = torch.zeros_like(logits)
        for target in range(node_count):
            positions = torch.nonzero(targets == target, as_tuple=False).flatten()
            if positions.numel():
                result[positions] = torch.softmax(logits[positions], dim=0)
        return result

    def forward(self, node_features: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        _validate_graph_inputs(node_features, edge_index, edge_features)
        edge_index = edge_index.to(device=node_features.device, dtype=torch.long)
        edge_features = edge_features.to(device=node_features.device, dtype=node_features.dtype)
        edge_index, edge_features = self._add_self_loops(
            node_features.shape[0], edge_index, edge_features
        )
        sources, targets = edge_index[0], edge_index[1]
        node_projected = self.node_projection(node_features).view(-1, self.heads, self.out_node_dim)
        edge_projected = self.edge_projection(edge_features).view(-1, self.heads, self.out_node_dim)
        score_input = torch.cat(
            [node_projected[sources], node_projected[targets], edge_projected], dim=-1
        )
        logits = torch.sum(score_input * self.attention.unsqueeze(0), dim=-1)
        logits = F.leaky_relu(logits, negative_slope=self.negative_slope)
        alpha = self._target_softmax(logits, targets, node_features.shape[0])

        messages = node_projected[sources] + edge_projected
        weighted_messages = messages * alpha.unsqueeze(-1)
        aggregated = torch.zeros_like(node_projected)
        aggregated.index_add_(0, targets, weighted_messages)
        if self.concat:
            output = aggregated.reshape(node_features.shape[0], self.heads * self.out_node_dim)
        else:
            output = aggregated.mean(dim=1)
        output = output + self.residual(node_features) + self.bias
        output = self.dropout(F.elu(output))
        self.last_attention = (edge_index.detach(), alpha.detach(), logits.detach())
        return output


class EdgeGATNetwork(nn.Module):
    """Stacked edge-aware GAT encoder returning one embedding per node."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        *,
        hidden_dim: int = 64,
        output_dim: int = 64,
        heads: int = 2,
        layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.output_dim = int(output_dim)
        self.layers = nn.ModuleList()
        if layers == 1:
            self.layers.append(
                EdgeGATLayer(
                    node_dim,
                    output_dim,
                    edge_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
            )
        else:
            self.layers.append(
                EdgeGATLayer(
                    node_dim,
                    hidden_dim,
                    edge_dim,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            current_dim = hidden_dim * heads
            for _ in range(layers - 2):
                self.layers.append(
                    EdgeGATLayer(
                        current_dim,
                        hidden_dim,
                        edge_dim,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                    )
                )
                current_dim = hidden_dim * heads
            self.layers.append(
                EdgeGATLayer(
                    current_dim,
                    output_dim,
                    edge_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
            )

    def forward(self, node_features: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        output = node_features
        for layer in self.layers:
            output = layer(output, edge_index, edge_features)
        return output

    def graph_embedding(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        *,
        root_index: int = 0,
    ) -> Tensor:
        """Return the local/root embedding used by the shared actor."""

        embeddings = self.forward(node_features, edge_index, edge_features)
        if embeddings.shape[0] == 0:
            raise ValueError("a graph must contain at least one node")
        if not 0 <= root_index < embeddings.shape[0]:
            raise ValueError("root_index is outside the graph")
        return embeddings[root_index]

    def attention_maps(self) -> list[tuple[Tensor, Tensor, Tensor] | None]:
        return [layer.last_attention for layer in self.layers]


__all__ = ["EdgeGATLayer", "EdgeGATNetwork"]
