import torch

def torch_int(x):
    """
    Minimal replacement for the old torch_int function.
    It should return True if the tensor has an integer dtype.
    """
    if isinstance(x, torch.Tensor):
        return x.dtype in (
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8,
        )
    return False
