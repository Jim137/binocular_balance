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
    collections: list,
    ax,
    neuron_index: int | list | None = None,
    is_box_plot: bool = False,
    is_mean_plot: bool = False,
    **kwargs
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
    if is_box_plot:
        ax = box_plot(activity, ax, **kwargs)
    elif is_mean_plot:
        ax = mean_plot(activity, ax, **kwargs)
    else:
        ax.plot(activity, **kwargs)
    return ax


def plot_weight_value(
    collections: list,
    ax,
    presynaptic_neuron_id: int | list | None = None,
    postsynaptic_neuron_id: int | list | None = None,
    is_box_plot: bool = False,
    is_mean_plot: bool = False,
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
    if is_box_plot:
        ax = box_plot(values, ax, **kwargs)
    elif is_mean_plot:
        ax = mean_plot(values, ax, **kwargs)
    else:
        ax.plot(values, **kwargs)
    return ax


def box_plot(seq, ax, **kwargs):
    if "num_box" in kwargs:
        num_box = kwargs["num_box"]
        del kwargs["num_box"]
    else:
        num_box = 10
    n = len(seq)
    box_size = int(np.ceil(n / num_box))
    boxes = []
    for i in range(num_box):
        boxes.append(seq[i * box_size : (i + 1) * box_size])
    ax.boxplot(boxes)
    ax.plot(np.arange(1, num_box + 1), [np.mean(box) for box in boxes], "r-", **kwargs)
    ax.set_xticklabels(np.arange(1, num_box + 1) * box_size)
    return ax


def mean_plot(seq, ax, **kwargs):
    num_cluster = len(seq) // 100
    boxes = []
    for i in range(100):
        boxes.append(seq[i * num_cluster : (i + 1) * num_cluster])
    ax.plot(np.arange(100) * num_cluster, [np.mean(box) for box in boxes], **kwargs)
    return ax
