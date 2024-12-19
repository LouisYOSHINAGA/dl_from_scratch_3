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


def sum_to(x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    ndim: int = len(shape)
    lead: int = x.ndim - ndim
    lead_axis: tuple[int, ...] = tuple(range(lead))

    axis: tuple[int, ...] = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y: np.ndarray = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

def reshape_sum_backward(gy: Variable, x_shape: tuple[int, ...], axis: int|tuple[int, ...]|None, keepdims: bool) -> Variable:
    ndim: int = len(x_shape)
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)
    else:
        tupled_axis = axis

    if ndim == 0 or tupled_axis is None or keepdims:
        shape = gy.shape
    else:
        actual_axis: list[int] = [axis if axis >= 0 else axis + ndim for axis in tupled_axis]
        shape: list[int] = list(gy.shape)
        for axis in sorted(actual_axis):
            shape.insert(axis, 1)
    return gy.reshape(shape)

def logsumexp(x: np.ndarray, axis: int|tuple[int, ...] =1) -> np.ndarray:
    m: np.ndarray = x.max(axis=axis, keepdims=True)
    y: np.ndarray = x - m
    np.exp(y, out=y)
    s: np.ndarray = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m


cache_dir: str = os.path.join(os.path.expanduser('~'), '.dezero')

def get_file(url: str, file_name: str|None =None) -> str:
    if file_name is None:
        file_name = url[url.rfind('/')+1:]
    file_path: str = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    if os.path.exists(file_path):
        return file_path

    print(f"Downloading: {file_name}")
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")
    return file_path

def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded: int = block_num * block_size
    p: float = downloaded / total_size * 100
    if p >= 100:
        p = 100
    i: int = int(downloaded / total_size * 30)
    if i >= 30:
        i = 30
    print(f"\r{'#'*i} {'.'*(30-i)} {p:.2f}%")

def pair(x: int|tuple[Any]) -> tuple[Any]:
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError


def get_conv_outsize(input_size: int, kernel_size: int, stride: int, pad: int) -> int:
    return (input_size + 2 * pad - kernel_size) // stride + 1

def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p