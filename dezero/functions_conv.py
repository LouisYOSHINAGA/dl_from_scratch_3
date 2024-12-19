import numpy as np
from dezero.core import Function, as_variable, Variable
from dezero.utils import pair, get_conv_outsize, get_deconv_outsize
from dezero.functions import linear, broadcast_to
from dezero.cuda import cupy, xpndarray, get_array_module


def im2col_array(img: xpndarray, kernel_size: int|tuple[int, int], stride: int|tuple[int, int],
                 pad: int|tuple[int, int], to_matrix: bool =True) -> xpndarray:
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH: int = get_conv_outsize(H, KH, SH, PH)
    OW: int = get_conv_outsize(W, KW, SW, PW)

    xp = get_array_module(img)
    if xp != np:
        col: xpndarray = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        img: np.ndarray = np.pad(img,
                                 ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                                 mode='constant', constant_values=(0,))
        col = np.array((N, C, KH, KW, OH, OW), dtype=img.dtype)

        for j in range(KH):
            j_lim: int = j + SH * OH
            for i in range(KW):
                i_lim: int = i + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))
    return col

def _im2col_gpu(img: xpndarray, kernel_size: int|tuple[int, int], stride: int|tuple[int, int],
                pad: int|tuple[int, int]) -> xpndarray:
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h: int = get_conv_outsize(h, kh, sy, ph)
    out_w: int = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col: xpndarray = cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col'
    )(img.reduced_view(), h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col

def col2im_array(col: int, img_shape: tuple[int,...], kernel_size: int|tuple[int, int], stride: int|tuple[int, int],
                 pad: int|tuple[int, int], to_matrix: bool =True) -> xpndarray:
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH: int = get_conv_outsize(H, KH, SH, PH)
    OW: int = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = get_array_module(col)
    if xp != np:
        img: xpndarray = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype)
        for j in range(KH):
            j_lim: int = j + SH * OH
            for i in range(KW):
                i_lim: int = i + SW * OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        return img[:, :, PH:H + PH, PW:W + PW]

def _col2im_gpu(col: int, sy: int, sx: int, ph: int, pw: int, h: int, w: int) -> xpndarray:
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img: xpndarray = cupy.empty((n, c, h, w), dtype=col.dtype)

    cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im'
    )(col.reduced_view(), h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img


class Im2col(Function):
    def __init__(self, kernel_size: int|tuple[int, int], stride: int|tuple[int, int],
                 pad: int|tuple[int, int], to_matrix: bool) -> None:
        super().__init__()
        self.input_shape: tuple[int,...]|None = None
        self.kernel_size: int|tuple[int, int] = kernel_size
        self.stride: int|tuple[int, int] = stride
        self.pad: int|tuple[int, int] = pad
        self.to_matrix: bool = to_matrix

    def forward(self, x: Variable|xpndarray) -> xpndarray:
        self.input_shape = x.shape
        return im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)

    def backward(self, gy: Variable) -> Variable:
        return col2im(gy, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

def im2col(x: Variable|xpndarray, kernel_size: int|tuple[int, int], stride: int|tuple[int, int] =1,
           pad: int|tuple[int, int] =0, to_matrix: bool =True) -> Variable:
    return Im2col(kernel_size, stride, pad, to_matrix)(x)


class Col2im(Function):
    def __init__(self, input_shape: tuple[int,...], kernel_size: int|tuple[int, int], stride: int|tuple[int, int],
                 pad: int|tuple[int, int], to_matrix: bool) -> None:
        super().__init__()
        self.input_shape: tuple[int,...] = input_shape
        self.kernel_size: int|tuple[int, int] = kernel_size
        self.stride: int|tuple[int, int] = stride
        self.pad: int|tuple[int, int] = pad
        self.to_matrix: bool = to_matrix

    def forward(self, x: Variable|xpndarray) -> xpndarray:
        return col2im_array(x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

    def backward(self, gy: Variable) -> Variable:
        return im2col(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)

def col2im(x: Variable, input_shape: tuple[int,...], kernel_size: int|tuple[int, int], stride: int|tuple[int, int] =1,
           pad: int|tuple[int, int] =0, to_matrix: bool =True) -> Variable:
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)


class Conv2d(Function):
    def __init__(self, stride: int|tuple[int, int] =1, pad: int|tuple[int, int] =0) -> None:
        super().__init__()
        self.stride: tuple[int, int] = pair(stride)
        self.pad: tuple[int, int] = pair(pad)

    def forward(self, x: xpndarray, W: xpndarray, b: xpndarray|None) -> xpndarray:
        xp = get_array_module(x)

        KH, KW = W.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)
        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))

        if b is not None:
            y += b
        return xp.rollaxis(y, 3, 1)

    def backward(self, gy: Variable) -> list[Variable|None]:
        x, W, b = self.inputs
        gx: Variable = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad, outsize=(x.shape[2], x.shape[3]))
        gW: Variable = Conv2DGradW(self)(x, gy)
        gb: Variable|None = None if b.data is None else gy.sum(axis=(0, 2, 3))
        return [gx, gW, gb]

def conv2d(x: Variable, W: Variable, b: Variable|None =None,
           stride: int|tuple[int, int] =1, pad: int|tuple[int, int] =0) -> Variable:
    return Conv2d(stride, pad)(x, W, b)

def conv2d_simple(x: xpndarray|Variable, W: Variable, b: Variable|None =None,
                  stride: int|tuple[int, int] =1, pad: int|tuple[int, int] =0) -> Variable:
    x = as_variable(x)
    W = as_variable(W)

    Weight: Variable = W
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH: int = get_conv_outsize(H, KH, SH, PH)
    OW: int = get_conv_outsize(W, KW, SW, PW)

    col: Variable = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    Weight: Variable = Weight.reshape(OC, -1).transpose()
    t: Variable = linear(col, Weight, b)
    return t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)


class Deconv2d(Function):
    def __init__(self, stride: int|tuple[int, int] =1, pad: int|tuple[int, int] =0,
                 outsize: int |None =None) -> None:
        super().__init__()
        self.stride: tuple[int, int] = pair(stride)
        self.pad: tuple[int, int] = pair(pad)
        self.outsize: int|None = outsize

    def forward(self, x: xpndarray, W: xpndarray, b: xpndarray|None) -> xpndarray:
        xp = get_array_module(x)

        Weight: xpndarray = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            out_h: int = get_deconv_outsize(H, KH, SH, PH)
            out_w: int = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape: tuple[int, int, int, int] = (N, OC, out_h, out_w)

        gcol: xpndarray = xp.tensordot(Weight, x, (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        y: xpndarray = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad, to_matrix=False)
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy: Variable) -> list[Variable|None]:
        x, W, b = self.inputs
        gx: Variable= conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        f = Conv2DGradW(self)
        gW: Variable = f(gy, x)
        gb: Variable|None = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return [gx, gW, gb]

def deconv2d(x: xpndarray|Variable, W: xpndarray|Variable, b: xpndarray|Variable|None =None,
             stride: int|tuple[int, int] =1, pad: int|tuple[int, int] =0, outsize: int|None =None) -> Variable:
    return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d: Conv2d) -> None:
        W: Variable = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size: tuple[int, int] = (kh, kw)
        self.stride: int|tuple[int, int] = conv2d.stride
        self.pad: int|tuple[int, int] = conv2d.pad

    def forward(self, x: xpndarray, gy: Variable) -> xpndarray:
        xp = get_array_module(x)
        col: xpndarray = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        return xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))

    def backward(self, gys: Variable) -> list[Variable]:
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad, outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return [gx, ggy]


class Pooling(Function):
    def __init__(self, kernel_size: int|tuple[int, int], stride: int|tuple[int, int] =1,
                 pad: int|tuple[int, int] =0) -> None:
        super().__init__()
        self.kernel_size: int|tuple[int, int] = kernel_size
        self.stride: int|tuple[int, int] = stride
        self.pad: int|tuple[int, int] = pad

    def forward(self, x: xpndarray) -> xpndarray:
        col: xpndarray = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes: xpndarray = col.argmax(axis=2)
        return col.max(axis=2)

    def backward(self, gy: Variable) -> Variable:
        return Pooling2DGrad(self)(gy)

def pooling(x: Variable, kernel_size: int|tuple[int, int], stride: int|tuple[int, int] =1,
           pad: int|tuple[int, int] =0) -> Variable:
    return Pooling(kernel_size, stride, pad)(x)

def pooling_simple(x: xpndarray|Variable, kernel_size: int|tuple[int, int],
                   stride: int|tuple[int, int] =1, pad: int|tuple[int, int] =0) -> Variable:
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col: Variable = im2col(x, kernel_size, stride, pad, to_matrix=True)
    col = col.reshape(-1, KH * KW)
    y: Variable = col.max(axis=1)
    return y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d: Pooling):
        self.mpool2d: Pooling = mpool2d
        self.kernel_size: int|tuple[int, int] = mpool2d.kernel_size
        self.stride: int|tuple[int, int] = mpool2d.stride
        self.pad: int|tuple[int, int] = mpool2d.pad
        self.input_shape: tuple[int, int, int, int] = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes: xpndarray = mpool2d.indexes

    def forward(self, gy: Variable) -> Variable:
        xp = get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        indexes: xpndarray = (self.indexes.ravel() + xp.arange(0, self.indexes.size * KH * KW, KH * KW))
        gcol: xpndarray = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)
        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)
        return col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride, self.pad, to_matrix=False)

    def backward(self, ggx: Variable) -> Variable:
        return Pooling2DWithIndexes(self.mpool2d)(ggx)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d: Pooling) -> None:
        self.kernel_size: int|tuple[int, int] = mpool2d.kernel_size
        self.stride: int|tuple[int, int] = mpool2d.stride
        self.pad: int|tuple[int, int] = mpool2d.pad
        self.input_shape: tuple[int, int, int, int] = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes: xpndarray = mpool2d.indexes

    def forward(self, x: xpndarray) -> xpndarray:
        col: xpndarray = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        indexes = self.indexes.ravel()
        col: xpndarray = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


class AveragePooling(Function):
    def __init__(self, kernel_size: int|tuple[int, int], stride: int|tuple[int, int] =1,
                 pad: int|tuple[int, int] =0) -> None:
        super().__init__()
        self.kernel_size: int|tuple[int, int] = kernel_size
        self.stride: int|tuple[int, int] = stride
        self.pad: int|tuple[int, int] = pad
        self.input_shape: tuple[int, int, int, int]|None = None

    def forward(self, x: xpndarray) -> xpndarray:
        self.input_shape: tuple[int, int, int, int] = x.shape
        col: xpndarray = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        return col.mean(axis=(2, 3))

    def backward(self, gy: Variable) -> Variable:
        N, C, OH, OW = gy.shape
        KW, KH = pair(self.kernel_size)
        gy /= (KW*KH)
        gcol: Variable = broadcast_to(gy.reshape(-1), (KH, KW, N*C*OH*OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        return col2im(gcol, self.input_shape, self.kernel_size, self.stride, self.pad, to_matrix=False)

def average_pooling(x: Variable, kernel_size: int|tuple[int, int],
                    stride: int|tuple[int, int] =1, pad: int|tuple[int, int] =0) -> Variable:
    return AveragePooling(kernel_size, stride, pad)(x)