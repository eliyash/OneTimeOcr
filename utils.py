import errno
import json
import os


def make_dir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
            

def save_data(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)


def load_data(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
        return data

