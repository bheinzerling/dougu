from torch import nn

from dougu import (
    Configurable,
    SubclassRegistry,
    WithLog,
    )


class ModelBase(Configurable, SubclassRegistry, WithLog, nn.Module):
    def __init__(self, conf):
        super().__init__(conf)
