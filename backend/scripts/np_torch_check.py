import numpy as np
import torch

print("numpy:", np.__version__)
print("torch:", torch.__version__)

try:
    a = np.zeros((2, 3), dtype=np.float32)
    t = torch.from_numpy(a)
    print("from_numpy_ok:", t.shape, t.dtype)
except Exception as e:
    print("from_numpy_error:", repr(e))

# Also verify torchvision nms availability again
try:
    import torchvision.ops as ops
    print("torchvision.ops.nms available:", hasattr(ops, "nms"))
except Exception as e:
    print("torchvision.ops import error:", repr(e))
