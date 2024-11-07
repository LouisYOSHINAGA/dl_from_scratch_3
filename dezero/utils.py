import os, subprocess
import urllib.request
import numpy as np
from dezero import Variable, Function
from typing import Callable, Any

IND: str = f"{' ' * 4}"
NWI: str = f"\n{IND}"  # newline with indent


def _dot_var(v: Variable, verbose: bool =False) -> str:
    name: str = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += f"{str(v.shape)} {str(v.dtype)}"
    return f"{id(v)} [label=\"{name}\", color=orange, style=filled]{NWI}"

def _dot_func(f: Function) -> Function:
    txt: str = f"{id(f)} [label=\"{f.__class__.__name__}\", color=lightblue, style=filled, shape=box]{NWI}"
    for x in f.inputs:
        txt += f"{IND}{id(x)} -> {id(f)}{NWI}"
    for y in f.outputs:
        txt += f"{IND}{id(f)} -> {id(y())}{NWI}"
    return txt

def get_dot_graph(output: Variable, verbose: bool =True) -> str:
    txt: str = ""
    funcs: list[Callable[[Any], Variable]] = []
    seen_set: set[Callable[[Any], Variable]] = set()

    def add_func(f: Function) -> None:
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func: Callable[[Any], Variable] = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)

    return f"digraph g{{{NWI}{txt[:-len(IND)]}}}"

def plot_dot_graph(output: Variable, verbose: bool =True, to_stdout: bool =False, to_file: str ="graph.png") -> None:
    tmp_dir: list[str] = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path: str = os.path.join(tmp_dir, "tmp_graph.dot")

    dot_graph: str = get_dot_graph(output, verbose)
    if to_stdout:
        print(f"{dot_graph}")
    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension: str = os.path.splitext(to_file)[1][1:]
    subprocess.run(f"dot {graph_path} -T {extension} -o {to_file}", shell=True)


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def sum_to(x, shape):
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m


def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')

cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')

def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + 2 * pad - kernel_size) // stride + 1

def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p