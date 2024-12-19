def get_conv_outsize(input_size: int, kernel_size: int, stride: int, pad: int) -> int:
    return (input_size + 2 * pad - kernel_size) // stride + 1


if __name__ == "__main__":
    insize_w: int = 4
    insize_h: int = 4
    kernel_w: int = 3
    kernel_h: int = 3
    stride_w: int = 1
    stride_h: int = 1
    padding_w: int = 1
    padding_h: int = 1

    outsize_w: int = get_conv_outsize(input_size=insize_w,
                                      kernel_size=kernel_w,
                                      stride=stride_w,
                                      pad=padding_w)
    outsize_h: int = get_conv_outsize(input_size=insize_h,
                                      kernel_size=kernel_h,
                                      stride=stride_h,
                                      pad=padding_h)
    print(f"outsize width = {outsize_w}, height = {outsize_h}")