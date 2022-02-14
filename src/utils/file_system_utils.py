import os
import json
import torch
from src.utils.model_helpers import model_getter


def create_folder_no_traceback(path_to_folder: str, rewrite: bool = False):
    """
    creates a folder, if rewrite, then clears the folder if it exists
    """
    if os.path.exists(path_to_folder) and os.path.isdir(path_to_folder):
        if not rewrite:
            return False
        clear_folder(path_to_folder)
        return True
    os.mkdir(path_to_folder)


def create_folder(path_to_folder: str, rewrite: bool = False):
    path = path_to_folder.split('/')
    p = ''
    for i in path:
        p += i
        if p!='':
            create_folder_no_traceback(p, rewrite)
        p += '/'


def clear_folder(path_to_folder: str):
    """
    clears the folder if exists
    """
    if os.path.exists(path_to_folder) and os.path.isdir(path_to_folder):
        for file in os.listdir(path_to_folder):
            os.remove(path_to_folder + "/" + file)
        return True
    return False


def save_model(model, path):
    # saving model params
    conf = model.get_params()
    with open(path+'/conf.json', 'w') as f:
        json.dump(conf, f)

    torch.save(model.state_dict(), path+'/model.pt')


def load_model(
    path: str
) -> torch.nn.Module:
    with open(path + '/conf.json', 'r') as f:
        conf = json.load(f)

    model_type = conf['name']
    del conf['name']

    model = model_getter(model_type, conf)
    model.load_state_dict(torch.load(path + '/model.pt'))
    return model
