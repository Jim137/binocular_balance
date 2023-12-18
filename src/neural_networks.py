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
    def _update(self):
        pass

    @abstractmethod
    def dynamic(self, time, is_record=False):
        pass


class nn(neural_network):
    def __init__(self, n_sensory: int, n_cortex: int, n_motor: int):
        super().__init__(n_sensory, n_cortex, n_motor)
        global id
        id = 0
        self.sensory = sensory(n_sensory)
        self.cortex = cortex(n_cortex)
        self.motor = motor(n_motor)
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

    def _update(self):
        self.sensory.update()
        self.cortex.update()
        self.motor.update()

    def dynamic(self, time, is_record=False):
        if is_record:
            recoding = []
        for _ in range(time):
            self._update()
            if is_record:
                recoding.append(self.record())
        if is_record:
            return recoding
        else:
            return None


class bisensory_nn(neural_network):
    def __init__(self, n_sensory: int, n_cortex: int, n_motor: int):
        super().__init__(n_sensory, n_cortex, n_motor)
        global id
        id = 0
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

    def _update(self):
        self.right_sensory.update()
        self.left_sensory.update()
        self.cortex.update()
        self.motor.update()

    def dynamic(self, time, is_record=False):
        if is_record:
            recoding = []
        for _ in range(time):
            self._update()
            if is_record:
                recoding.append(self.record())
        if is_record:
            return recoding
        else:
            return None
