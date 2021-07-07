import os


def create_folder(path_to_folder, rewrite=False):
    if os.path.exists(path_to_folder) and os.path.isdir(path_to_folder):
        if not rewrite:
            return False
        clear_folder(path_to_folder)
        return True
    os.mkdir(path_to_folder)


def clear_folder(path_to_folder):
    if os.path.exists(path_to_folder) and os.path.isdir(path_to_folder):
        for file in os.listdir(path_to_folder):
            os.remove(path_to_folder + '/' + file)
        return True
    return False
