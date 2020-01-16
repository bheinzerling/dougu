import torch

from .metric import Metric


class MeanReciprocalRank(Metric):

    def __init__(self, *args, mode='prob', **kwargs):
        super().__init__(*args, **kwargs)
        self.update = {
            'prob': self.update_prob,
            'idx': self.update_idx}[mode]

    def reset(self, mode='prob'):
        self.ranks = []

    def update(self, output):
        raise NotImplementedError()

    def update_prob(self, output):
        probs, target = output
        if (probs < 0).any():
            assert (probs <= 0).all()
            probs = torch.exp(probs)
        correct_prob = probs.gather(1, target.unsqueeze(1))
        rank = (probs >= correct_prob).sum(dim=1)
        self.ranks.append(rank.cpu())

    def update_idx(self, output):
        probs, target = output
        breakpoint()

    def compute(self):
        rank = torch.cat(self.ranks)
        return (1 / rank.float()).mean()
