import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        # out_features: int,
        num_projections: int,
        loss_scale: float = 3e-6,
    ) -> None:
        super().__init__()

        self.num_projections = num_projections
        self.in_features = in_features
        # self.out_features = out_features
        self.loss_scale = loss_scale

        self.projection_layers = nn.ModuleList(
            [
                # nn.Linear(in_features=in_features, out_features=out_features)
                nn.Linear(in_features=in_features, out_features=in_features)
                for _ in range(num_projections)
            ]
        )
        self.router = nn.Linear(in_features=in_features, out_features=num_projections)

    def compute_loss(
        self,
        layer: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:  # TODO: Check the loss function with the loss
        fractions = torch.zeros(self.num_projections)
        probability = torch.zeros(self.num_projections)
        T_i = layer.shape[0]

        for i in range(self.num_projections):
            fractions[i] = torch.sum(layer == i) / T_i
            probability[i] = (
                torch.sum(probs[layer == i].max(dim=1)[0] / layer.shape[0]) / T_i
            )

        aux_loss = (
            torch.sum(fractions * probability) * self.loss_scale * self.num_projections
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
        for i in range(self.num_projections):
            output[layer == i] = self.projection_layers[i](x[layer == i]) * probs[
                layer == i
            ].max(dim=1)[0].unsqueeze(1)

        output = output.reshape(B, C, -1)
        return output, loss


if __name__ == "__main__":
    a = torch.rand(64, 22, 256)
    layer = MoELayer(256, 4)
    logits, loss = layer(a)
    print(loss)
    pass
