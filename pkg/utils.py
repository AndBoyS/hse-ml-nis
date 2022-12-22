from typing import Tuple

import torch


def prune_linear_layer(
        model: torch.nn.Module,
        neurons_to_keep: Tuple[int, ...],
        layer_name: str = 'classifier',
        ):
    """
    Оставить только указанные нейроны в линейном слое

    model: torch.nn.Module
        Модель с линейным слоев
    neurons_to_keep: Tuple[int]
        Айди нейронов, которые надо оставить
    layer_name: str
        Название линейного слоя, default 'classifier'
    """

    old_linear = getattr(model, layer_name)
    in_features = old_linear.in_features
    out_features = len(neurons_to_keep)

    old_weight_val = model.classifier.weight[list(neurons_to_keep)]
    new_linear = torch.nn.Linear(in_features, out_features)

    with torch.no_grad():
        new_linear.weight[:] = old_weight_val
        setattr(model, layer_name, new_linear)
