import numpy as np


id = 0


class Neuron:
    def __init__(self):
        global id
        self.value = np.float64(0)
        self.presynaptic_neuron = []
        self.weights = []
        self.bias = np.float64(0)
        self.input = np.float64(0)
        self.input_fluctuation_rate = None
        self.timestamp = 0
        self.tag = None
        self.id = id
        id += 1

    def __call__(self):
        return self.value

    def update(
        self,
        gain=lambda x: np.max([x, np.float64(0)]),
    ):
        sum = [
            weight.value * weight.presynaptic_neuron.value for weight in self.weights
        ]
        if self.input_fluctuation_rate:
            sum.append(
                np.random.normal(
                    loc=self.input,
                    scale=self.input_fluctuation_rate,
                )
            )
        else:
            sum.append(self.input)
        self.value = gain(np.sum(sum))
        self.timestamp += 1

    def add_presynaptic_neuron(self, neuron):
        self.presynaptic_neuron.append(neuron)
        self.weights.append(weight(neuron, self))

    def weights_metadata(self):
        return [weight.metadata() for weight in self.weights]

    def metadata(self):
        return {
            "value": self.value,
            "presynaptic_neuron": self.presynaptic_neuron,
            "weights": self.weights_metadata(),
            "bias": self.bias,
            "input": self.input,
            "timestamp": self.timestamp,
            "tag": self.tag,
            "id": self.id,
        }


class weight:
    def __init__(self, pre: Neuron, post: Neuron):
        self.value = np.float128(1e-1)
        self.presynaptic_neuron = pre
        self.postsynaptic_neuron = post
        self.threshold = np.float64(0)
        self.timestamp = 0

    def __call__(self):
        return self.value

    def _threshold(self, th_time_constant=1e3):
        self.threshold *= np.exp(-1 / th_time_constant)
        self.threshold += self.postsynaptic_neuron.value / th_time_constant

    def update(self, learning_rate=0.1, method="cocktail", th_time_constant=1e3):
        if method == "hebbian" or method == "hebb":
            tmp = np.float128(
                self.postsynaptic_neuron.value * self.presynaptic_neuron.value
                - self.value * np.square(self.postsynaptic_neuron.value)
            )
        elif method == "bcm":
            self._threshold(th_time_constant)
            tmp = np.float128(
                self.postsynaptic_neuron.value
                * (self.postsynaptic_neuron.value - self.threshold)
                * self.presynaptic_neuron.value
            )
        elif method == "cocktail":
            self._threshold(th_time_constant)
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
        num_cluster = int(np.ceil(n_data / self.number_of_neurons))
        for neuron in self.neurons:
            neuron.input = np.float64(0)
        for i in range(n_data):
            j = i // num_cluster
            self.neurons[j].input += data[i] / num_cluster

    def update(self, learning_rate=0.1, method="cocktail"):
        for neuron in self.neurons:
            neuron.update()
            for weight in neuron.weights:
                weight.update(learning_rate, method=method)

    def collect(self):
        collection = [neuron.metadata() for neuron in self.neurons]
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

    def update(self, learning_rate=0.1, method="cocktail"):
        for neuron in self.neurons:
            neuron.update()
            for weight in neuron.weights:
                weight.update(learning_rate, method=method)

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

    def update(self, learning_rate=0.1, method="cocktail"):
        for neuron in self.neurons:
            neuron.update()
            for weight in neuron.weights:
                weight.update(learning_rate, method=method)

    def collect(self):
        return [neuron.metadata() for neuron in self.neurons]
