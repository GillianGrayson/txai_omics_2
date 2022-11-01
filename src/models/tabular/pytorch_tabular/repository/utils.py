import logging
import torch.nn as nn


logger = logging.getLogger(__name__)


def _initialize_layers(activation, initialization, layers):
    if type(layers) == nn.Sequential:
        for layer in layers:
            if hasattr(layer, "weight"):
                _initialize_layers(activation, initialization, layer)
    else:
        if activation == "ReLU":
            nonlinearity = "relu"
        elif activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            if initialization == "kaiming":
                logger.warning(
                    "Kaiming initialization is only recommended for ReLU and LeakyReLU."
                )
                nonlinearity = "leaky_relu"
            else:
                nonlinearity = "relu"

        if initialization == "kaiming":
            nn.init.kaiming_normal_(layers.weight, nonlinearity=nonlinearity)
        elif initialization == "xavier":
            nn.init.xavier_normal_(
                layers.weight,
                gain=nn.init.calculate_gain(nonlinearity)
                if activation in ["ReLU", "LeakyReLU"]
                else 1,
            )
        elif initialization == "random":
            nn.init.normal_(layers.weight)


def _linear_dropout_bn(activation, initialization, use_batch_norm, in_units, out_units, dropout):
    if isinstance(activation, str):
        _activation = getattr(nn, activation)
    else:
        _activation = activation
    layers = []
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(num_features=in_units))
    linear = nn.Linear(in_units, out_units)
    _initialize_layers(activation, initialization, linear)
    layers.extend([linear, _activation()])
    if dropout != 0:
        layers.append(nn.Dropout(dropout))
    return layers
