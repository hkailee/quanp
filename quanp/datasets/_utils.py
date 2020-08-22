from functools import wraps

from .._settings import settings


def check_datasetdir_exists(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        settings.datasetdir.mkdir(exist_ok=True)
        return f(*args, **kwargs)

    return wrapper

def check_logdir_exists(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        settings.logdir.mkdir(exist_ok=True)
        return f(*args, **kwargs)

    return wrapper
