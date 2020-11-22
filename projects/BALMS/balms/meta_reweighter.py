import torch.nn as nn


def init_weights_zero(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, val=0)


class Learner(nn.Module):
    """
    Meta Re-weight
    """
    def __init__(self, num_classes):
        super(Learner, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_classes, 1, bias=False),
            nn.Sigmoid()
        )

        # initialize weight to 0.5
        self.fc.apply(init_weights_zero)

    def forward(self, x, targets):
        targets = targets.float()
        weighted_onehot = self.fc(targets).squeeze(-1)
        x = weighted_onehot * x
        x = x / weighted_onehot.sum() * targets.sum()
        return x