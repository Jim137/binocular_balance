from abc import ABCMeta, abstractmethod

from .neuron import sensory, cortex, motor


class neural_network(object, metaclass=ABCMeta):
    def __init__(self, n_sensory: int, n_cortex: int, n_motor: int):
        self.n_sensory = n_sensory
        self.n_cortex = n_cortex
        self.n_motor = n_motor

    @abstractmethod
    def add_input(self, data):
        pass

    @abstractmethod
    def record(self):
        pass

    @abstractmethod
    def _update(self, learning_rate=0.1, method="cocktail"):
        pass

    @abstractmethod
    def dynamic(self, time, learning_rate=0.1, method="cocktail", is_record=False):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class nn(neural_network):
    def __init__(
        self, n_sensory: int, n_cortex: int, n_motor: int, is_cortex_fully_connect=False
    ):
        super().__init__(n_sensory, n_cortex, n_motor)
        self.sensory = sensory(n_sensory)
        self.cortex = cortex(n_cortex)
        self.motor = motor(n_motor)
        if is_cortex_fully_connect:
            self.cortex.fully_connect()
        self.cortex.add_sensory(self.sensory)
        self.motor.add_cortex(self.cortex)

    def add_input(self, data):
        self.sensory.input(data)

    def record(self):
        collection = {}
        collection["sensory"] = self.sensory.collect()
        collection["cortex"] = self.cortex.collect()
        collection["motor"] = self.motor.collect()
        return collection

    def _update(self, learning_rate=0.1, method="cocktail"):
        self.sensory.update(learning_rate, method)
        self.cortex.update(learning_rate, method)
        self.motor.update(learning_rate, method)

    def dynamic(self, time, learning_rate=0.1, method="cocktail", is_record=False):
        if is_record:
            recoding = []
        for _ in range(time):
            self._update(learning_rate, method)
            if is_record:
                recoding.append(self.record())
        if is_record:
            return recoding
        else:
            return None

    def __iter__(self):
        neurons = []
        neurons.extend(self.sensory.neurons)
        neurons.extend(self.cortex.neurons)
        neurons.extend(self.motor.neurons)
        return iter(neurons)


class bisensory_nn(neural_network):
    def __init__(self, n_sensory: int, n_cortex: int, n_motor: int):
        super().__init__(n_sensory, n_cortex, n_motor)
        self.right_sensory = sensory(n_sensory)
        self.left_sensory = sensory(n_sensory)
        self.cortex = cortex(n_cortex)
        self.motor = motor(n_motor)
        self.cortex.fully_connect()
        self.cortex.add_sensory(self.right_sensory)
        self.cortex.add_sensory(self.left_sensory)
        self.motor.add_cortex(self.cortex)

    def add_input(self, data0, data1=None):
        if data1 is None:
            data1 = data0
        self.right_sensory.input(data0)
        self.left_sensory.input(data1)

    def record(self):
        collection = {}
        collection["right_sensory"] = self.right_sensory.collect()
        collection["left_sensory"] = self.left_sensory.collect()
        collection["cortex"] = self.cortex.collect()
        collection["motor"] = self.motor.collect()
        return collection

    def _update(self, learning_rate=0.1, method="cocktail"):
        self.right_sensory.update(learning_rate, method)
        self.left_sensory.update(learning_rate, method)
        self.cortex.update(learning_rate, method)
        self.motor.update(learning_rate, method)

    def dynamic(self, time, learning_rate=0.1, method="cocktail", is_record=False):
        if is_record:
            recoding = []
        for _ in range(time):
            self._update(learning_rate, method)
            if is_record:
                recoding.append(self.record())
        if is_record:
            return recoding
        else:
            return None

    def __iter__(self):
        neurons = []
        neurons.extend(self.right_sensory.neurons)
        neurons.extend(self.left_sensory.neurons)
        neurons.extend(self.cortex.neurons)
        neurons.extend(self.motor.neurons)
        return iter(neurons)
