import torch
from src.networks.modules import LSTMMultiplePointProcesses


def model_getter(model_type: str, model_params: dict):
    if model_type == "LSTM":
        return LSTMMultiplePointProcesses(**model_params)
    else:
        raise Exception("Unknown model type")
