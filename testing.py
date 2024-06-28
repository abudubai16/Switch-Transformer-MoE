import torch
import torch.nn as nn
import torch.nn.functional as F


class MultipleProjections(nn.Module):
    def __init__(
        self,
        in_features: int,
        # out_features: int,
        num_projections: int,
        loss_scale: float = 3e-6,
    ) -> None:
        super().__init__()

        self.num_projections = num_projections
        # self.out_features = out_features
        self.loss_scale = loss_scale

        self.projection_layers1 = nn.ModuleList(
            [
                # nn.Linear(in_features=in_features, out_features=out_features)
                nn.Linear(in_features=in_features, out_features=in_features)
                for _ in range(num_projections)
            ]
        )
        self.router = nn.Linear(in_features=in_features, out_features=num_projections)

    def compute_loss(
        self, layer: torch.Tensor
    ) -> torch.Tensor:  # TODO: Check the loss function with the loss
        a = torch.zeros(self.num_projections)
        for i in range(self.num_projections):
            a[i] = (layer == i).sum()
        variation = a - a.mean()
        loss = (variation**2).mean()
        return loss * self.loss_scale

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq, d_model)
        """
        assert x.ndim == 3

        B, C, D = x.shape

        x = x.reshape(B * C, -1)
        layer = self.router(x).argmax(dim=1)
        print(layer)
        exit()
        loss = self.compute_loss(layer=layer)

        output = torch.zeros(B * C, self.out_features, device="cuda")
        for i in range(self.num_projections):
            output[layer == i] = self.projection_layers1[i](x[layer == i])

        output = output.reshape(B, C, -1)
        return output, loss


if __name__ == "__main__":
    in_feature = 256
    num_projections = 4

    layer = MultipleProjections(in_features=in_feature, num_projections=num_projections)
    pass
