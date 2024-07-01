import torch
import torch.nn as nn
import torch.nn.functional as F


def init_w(layer):
    assert isinstance(
        layer, torch.Tensor
    ), f"The tensor must be an instance of a torch.Tensor, instead its a type {type(layer)}"
    layer = nn.init.xavier_normal_(layer)
    return layer


class MoELayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        dim_feedforward: int,
        num_experts: int,
        activation: nn.functional = F.relu,
        dropout1: nn.Dropout = nn.Dropout(),
        dropout2: nn.Dropout = nn.Dropout(),
        loss_scale: float = 3e-6,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.in_features = in_features
        self.loss_scale = loss_scale

        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.activation = activation

        w1 = torch.empty(num_experts, in_features, dim_feedforward)
        w2 = torch.empty(num_experts, dim_feedforward, in_features)

        self.w1 = nn.Parameter(init_w(w1))
        self.w2 = nn.Parameter(init_w(w2))
        self.linear1 = lambda pos, mat: mat @ self.w1[pos]
        self.linear2 = lambda pos, mat: mat @ self.w2[pos]

        self.router = nn.Linear(in_features=in_features, out_features=num_experts)

    def compute_loss(
        self,
        layer: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        fractions = torch.zeros(self.num_experts)
        probability = torch.zeros(self.num_experts)
        T_i = layer.shape[0]

        for i in range(self.num_experts):
            fractions[i] = torch.sum(layer == i) / T_i
            probability[i] = (
                torch.sum(probs[layer == i].max(dim=1)[0] / layer.shape[0]) / T_i
            )

        aux_loss = (
            torch.sum(fractions * probability) * self.loss_scale * self.num_experts
        )
        return aux_loss

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq, d_model)
        """
        assert x.ndim == 3

        B, C, D = x.shape
        x = x.reshape(B * C, -1)

        probs = F.softmax(self.router(x), dim=1)
        layer = probs.argmax(dim=1)
        loss = self.compute_loss(layer=layer, probs=probs)

        output = torch.zeros(B * C, self.in_features)
        for i in range(self.num_experts):
            prob = probs[layer == i].max(dim=1)[0].unsqueeze(1)
            values = x[layer == i]
            logits = self.linear2(
                i, self.dropout1(self.activation(self.linear1(i, values)))
            )
            output[layer == i] = self.dropout2(logits) * prob

        output = output.reshape(B, C, -1)
        return output, loss


if __name__ == "__main__":
    pass
