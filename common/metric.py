import torch

class MetricBase(object):
    def __init__(self, name):
        self.name = str(name)
        self.reset()
    
    def __str__(self):
        return 'EvalMetric: {}'.format(dict(self.get_name_value()))

    def get_name_value(self):
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    def update(self, outputs):
        raise NotImplementedError()

    def reset(self):
        self.num_inst = torch.tensor(0.)
        self.sum_metric = torch.tensor(0.)

    def get(self):
        if self.num_inst.item() == 0:
            return (self.name, float('nan'))
        else:
            metric_tensor = (self.sum_metric / self.num_inst).detach().cpu()
        return (self.name, metric_tensor.item())