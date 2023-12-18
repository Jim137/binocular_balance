import numpy as np


global id
id = 0


class Neuron:
    def __init__(self):
        global id
        self.value = np.float32(0)
        self.presynaptic_neuron = []
        self.weights = []
        self.bias = np.float32(0)
        self.input = np.float32(0)
        self.timestamp = 0
        self.tag = None
        self.id = id
        id += 1

    def __call__(self):
        return self.value

    def update(self, gain=lambda x: np.max([x, np.float32(0)])):
        sum = [
            weight.value * weight.presynaptic_neuron.value for weight in self.weights
        ]
        sum.append(self.input)
        self.value = gain(np.sum(sum))
        self.timestamp += 1

    def add_presynaptic_neuron(self, neuron):
        self.presynaptic_neuron.append(neuron)
        self.weights.append(weight(neuron, self))

    def metadata(self):
        return {
            "value": self.value,
            "presynaptic_neuron": self.presynaptic_neuron,
            "weights": self.weights,
            "bias": self.bias,
            "input": self.input,
            "timestamp": self.timestamp,
            "tag": self.tag,
            "id": self.id,
        }


class weight:
    def __init__(self, pre: Neuron, post: Neuron):
        self.value = np.float32(1e-5)
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

    def update(self, learning_rate=0.1, method="cocktail"):
        self._threshold()
        if method == "hebbian" or method == "hebb":
            tmp = np.float128(
                self.postsynaptic_neuron.value * self.presynaptic_neuron.value
                - self.value * np.square(self.postsynaptic_neuron.value)
            )
        elif method == "bcm":
            tmp = np.float128(
                self.postsynaptic_neuron.value
                * (self.postsynaptic_neuron.value - self.threshold)
                * self.presynaptic_neuron.value
            )
        elif method == "cocktail":
            tmp = np.float128(
                self.postsynaptic_neuron.value
                * (self.postsynaptic_neuron.value - self.threshold)
                * self.presynaptic_neuron.value
                - self.value * np.square(self.postsynaptic_neuron.value)
            )
        else:
            raise ValueError("method must be hebbian, bcm or cocktail")
        self.value += learning_rate * tmp

    def metadata(self):
        return {
            "value": self.value,
            "presynaptic_neuron_id": self.presynaptic_neuron.id,
            "postsynaptic_neuron_id": self.postsynaptic_neuron.id,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


class sensory:
    def __init__(self, number_of_neurons: int):
        self.number_of_neurons = number_of_neurons
        self.neurons = [Neuron() for _ in range(number_of_neurons)]
        for neuron in self.neurons:
            neuron.tag = "sensory"

    def input(self, data):
        n_data = len(data)

        # pseudo neurons here act as convolutional neurons
        self.pseudo_neurons = [Neuron() for _ in range(n_data)]
        for i, neuron in enumerate(self.pseudo_neurons):
            neuron.tag = "pseudo_sensory"
            neuron.input = data[i]

        num_cluster = int(n_data / self.number_of_neurons) + 1
        for i in range(n_data):
            j = i // num_cluster
            self.neurons[j].add_presynaptic_neuron(self.pseudo_neurons[i])

    def update(self, learning_rate=0.1):
        for neuron in self.pseudo_neurons:
            neuron.update()
        for neuron in self.neurons:
            neuron.update()
            for weight in neuron.weights:
                weight.update(learning_rate)

    def collect(self, is_collect_pseudo=False):
        collection = [neuron.metadata() for neuron in self.neurons]
        if is_collect_pseudo:
            collection.extend([neuron.metadata() for neuron in self.pseudo_neurons])
        return collection


class cortex:
    def __init__(self, number_of_neurons: int):
        self.number_of_neurons = number_of_neurons
        self.neurons = [Neuron() for _ in range(number_of_neurons)]
        for neuron in self.neurons:
            neuron.tag = "cortex"

    def fully_connect(self):
        for i in range(self.number_of_neurons):
            for j in range(self.number_of_neurons):
                if i == j:
                    continue
                self.neurons[i].add_presynaptic_neuron(self.neurons[j])

    def add_sensory(self, sensory: sensory):
        for i in range(self.number_of_neurons):
            for j in range(sensory.number_of_neurons):
                self.neurons[i].add_presynaptic_neuron(sensory.neurons[j])

    def update(self, learning_rate=0.1):
        for neuron in self.neurons:
            neuron.update()
            for weight in neuron.weights:
                weight.update(learning_rate)

    def collect(self):
        return [neuron.metadata() for neuron in self.neurons]


class motor:
    def __init__(self, number_of_neurons: int):
        self.number_of_neurons = number_of_neurons
        self.neurons = [Neuron() for _ in range(number_of_neurons)]
        for neuron in self.neurons:
            neuron.tag = "motor"

    def add_cortex(self, cortex: cortex):
        for i in range(self.number_of_neurons):
            for j in range(cortex.number_of_neurons):
                self.neurons[i].add_presynaptic_neuron(cortex.neurons[j])

    def update(self, learning_rate=0.1):
        for neuron in self.neurons:
            neuron.update()
            for weight in neuron.weights:
                weight.update(learning_rate)

    def collect(self):
        return [neuron.metadata() for neuron in self.neurons]
