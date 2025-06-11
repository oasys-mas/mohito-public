import torch
import torch.nn as nn


class Usqueezer(nn.Module):

    def __init__(self, index: int, do_unsqueeze: bool, *args, **kwargs):
        super(Usqueezer, self).__init__()
        self.index = index
        self.do_unsqueeze = do_unsqueeze

    def forward(self, x: torch.Tensor):
        if self.do_unsqueeze:
            return x.unsqueeze(self.index)
        else:
            return x.squeeze(self.index)