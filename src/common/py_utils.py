import logging
import os


class NewLineFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\n" + parts[0])

        return msg


def unique_experiment_name(config):
    configs = "_".join(
        [os.path.splitext(os.path.basename(p))[0] for p in config.config_filenames]
    )
    database_name = config.dataset.name

    unique_name = f"{configs}___{database_name}"

    return unique_name


def get_num_total_lines(p):
    import subprocess

    # result = subprocess.run(['wc', '-l', p], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if not isinstance(p, str):
        p = str(p)

    result = subprocess.check_output(["wc", "-l", p])
    if isinstance(result, bytes):
        result = result.decode("utf-8")

    return int(result.strip().split(" ")[0])


def softmax(X, theta=1.0, axis=None):
    """
    Copyright: https://nolanbconaway.github.io/blog/2017/softmax-numpy.html
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    import numpy as np

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p
