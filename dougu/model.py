from torch import nn

from dougu import (
    Configurable,
    SubclassRegistry,
    )


class ModelBase(Configurable, SubclassRegistry, nn.Module):
    def __init__(self, conf):
        super().__init__(conf)
