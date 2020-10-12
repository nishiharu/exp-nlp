import torch
import importlib

class Reader(torch.nn.Module):
    def __init__(self, config):
        super(Reader, self).__init__()
        raise NotImplementedError()

    def forward(self, x):
        return x

class Programmer(torch.nn.Module):
    def __init__(self, config):
        super(Programmer, self).__init__()
        raise NotImplementedError()

    def forward(self, x):
        return x

class ExpNet(torch.nn.Module):
    def __init__(self, config):
        super(ExpNet, self).__init__()
        self.embedding = \
            importlib.import_module(config.TASK).embedding.Embedding(config)
        self.reader = Reader(config)
        self.programmer = Programmer(config)

    def forward(self, x):
        return x