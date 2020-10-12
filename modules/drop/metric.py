from common.metric import MetricBase

class Metric(MetricBase):
    def update(self, output):
        raise NotImplementedError()