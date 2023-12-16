from typing import Any
import numpy as np


class Neuron:
    def __init__(self):
        self.value = np.float32(0)
        self.presynaptic_neuron = []
        self.weights = []
        self.bias = 0

    def __call__(self):
        return self.value

    def update(self, gain=lambda x: np.max([x, np.float32(0)])):
        self.value = gain(
            np.sum(
                [
                    weight.value * weight.presynaptic_neuron.value
                    for weight in self.weights
                ]
            )
        )

    def add_presynaptic_neuron(self, neuron):
        self.presynaptic_neuron.append(neuron)
        self.weights.append(weight(neuron, self))


class weight:
    def __init__(self, pre: Neuron, post: Neuron):
        self.value = np.float32(0)
        self.presynaptic_neuron = pre
        self.postsynaptic_neuron = post
        self.threshold = np.float32(0)
        self.timestamp = 0

    def __call__(self):
        return self.value

    def _threshold(self):
        self.threshold = (
            self.threshold * self.timestamp + np.square(self.postsynaptic_neuron.value)
        ) / (self.timestamp + 1)
        self.timestamp += 1

    def update(self, learning_rate=0.1):
        tmp = self.postsynaptic_neuron.value * (
            self.postsynaptic_neuron.value - self.threshold + 1
        ) * self.presynaptic_neuron.value - self.value * np.square(
            self.postsynaptic_neuron.value
        )
        self.value = self.value + learning_rate * tmp
        self._threshold()
