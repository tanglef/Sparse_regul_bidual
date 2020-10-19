import os

def save_fig(path, name, extension):
    name = name + '.' + extension
    return os.path.join(path, name)
