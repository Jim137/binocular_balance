import numpy as np
import matplotlib.pyplot as plt


def record_splitter(recording: list, nn_type: str):
    if nn_type == "bisensory":
        sensory = [[], []]
    else:
        sensory = []
    cortex = []
    motor = []
    for collection in recording:
        if nn_type == "bisensory":
            sensory[0].append(collection["right_sensory"])
            sensory[1].append(collection["left_sensory"])
        else:
            sensory.append(collection["sensory"])
        cortex.append(collection["cortex"])
        motor.append(collection["motor"])
    return sensory, cortex, motor


def plot_neuron_activity(
    collections: list, ax, neuron_index: int | list | None = None, **kwargs
):
    """
    If neuron_index is None, plot the average activity of all neurons.
    """
    if type(neuron_index) == int:
        neuron_index = [neuron_index]

    activity = []
    for collection in collections:
        if neuron_index is None:
            activity.append(np.mean([neuron["value"] for neuron in collection]))
        else:
            tmp = []
            for neuron in collection:
                if neuron["id"] in neuron_index:
                    tmp.append(neuron["value"])
            activity.append(np.mean(tmp))
    ax.plot(activity, **kwargs)
    return ax


def plot_weight_value(
    collections: list,
    ax,
    presynaptic_neuron_id: int | list | None = None,
    postsynaptic_neuron_id: int | list | None = None,
    **kwargs
):
    """
    If presynaptic_neuron_id or postsynaptic_neuron_id is None, plot the average weight value of missings arguments.
    """
    if type(presynaptic_neuron_id) == int:
        presynaptic_neuron_id = [presynaptic_neuron_id]
    if type(postsynaptic_neuron_id) == int:
        postsynaptic_neuron_id = [postsynaptic_neuron_id]

    values = []
    for collection in collections:
        if presynaptic_neuron_id is None and postsynaptic_neuron_id is None:
            values.append(
                np.mean(
                    [
                        weight["value"]
                        for neuron in collection
                        for weight in neuron["weights"]
                    ]
                )
            )
        elif presynaptic_neuron_id is None:
            tmp = []
            for neuron in collection:
                if neuron["id"] in postsynaptic_neuron_id:
                    tmp.extend([weight["value"] for weight in neuron["weights"]])
            values.append(np.mean(tmp))
        elif postsynaptic_neuron_id is None:
            tmp = []
            for neuron in collection:
                for weight in neuron["weights"]:
                    if weight["presynaptic_neuron_id"] in presynaptic_neuron_id:
                        tmp.append(weight["value"])
            values.append(np.mean(tmp))
        else:
            tmp = []
            for neuron in collection:
                if neuron["id"] in postsynaptic_neuron_id:
                    for weight in neuron["weights"]:
                        if weight["presynaptic_neuron_id"] in presynaptic_neuron_id:
                            tmp.append(weight["value"])
            values.append(np.mean(tmp))
    ax.plot(values, **kwargs)
    return ax
