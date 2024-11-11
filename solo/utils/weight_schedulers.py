from typing import Any


class TriangleScheduler():
    def __init__(self, start_weight, max_weight, end_weight, start_epoch, mid_epoch, end_epoch):
        self.start_weight = start_weight
        self.max_weight = max_weight
        self.end_weight = end_weight
        self.start_epoch = start_epoch
        self.mid_epoch = mid_epoch
        self.end_epoch = end_epoch

    def __call__(self, epoch):
        if epoch < self.start_epoch:
            return self.start_weight
        elif epoch < self.mid_epoch:
            slope = (self.max_weight - self.start_weight) / (self.mid_epoch - self.start_epoch)
            return slope * (epoch - self.start_epoch) + self.start_weight
        elif epoch < self.end_epoch:
            slope = (self.end_weight - self.max_weight) / (self.end_epoch - self.mid_epoch)
            return slope * (epoch - self.mid_epoch) + self.max_weight
        else:
            return self.end_weight


class WarmupScheduler():
    def __init__(self, base_weight, warmup_epochs, weight, reg_epochs):
        self.base_weight = base_weight
        self.warmup_epochs = warmup_epochs
        self.weight = weight
        self.reg_epochs = reg_epochs

    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            return self.base_weight
        elif epoch < self.reg_epochs + self.warmup_epochs:
            return self.weight
        else:
            return self.base_weight

class StepScheduler():
    def __init__(self, weight, steps: list[int], scale = 0.1, warmup_epochs=5):
        self.weight = weight
        self.steps = steps
        self.scale = scale
        self.warmup_epochs = warmup_epochs

    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            return 0
        while self.steps and epoch >= self.steps[0]:
            self.weight *= self.scale
            self.steps.pop(0)
        return self.weight

class IntervalScheduler():
    def __init__(self, intervals: list[list[int]], max_epochs) -> None:
        self.max_epochs = max_epochs
        self.weight_per_epoch = [0 for _ in range(max_epochs)]
        for (x, y, weight) in intervals:
            for i in range(x, y):
                self.weight_per_epoch[i] = weight

    def __call__(self, epoch) -> float:
        if epoch > self.max_epochs:
            return 0
        return self.weight_per_epoch[epoch]

class ConstantScheduler():
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, epoch):
        return self.weight
