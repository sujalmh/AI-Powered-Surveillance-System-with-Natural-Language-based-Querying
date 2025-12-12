import sys
print("Python:", sys.version)

def safe_import(name):
    try:
        m = __import__(name)
        return m, None
    except Exception as e:
        return None, e

torch, e_torch = safe_import("torch")
print("torch import ok:", e_torch is None)
if torch:
    print("torch.__version__:", getattr(torch, "__version__", "NA"))
    print("torch.cuda.is_available():", torch.cuda.is_available() if hasattr(torch, "cuda") else "NA")
    try:
        print("torch.version.cuda:", getattr(torch.version, "cuda", "NA"))
    except Exception:
        print("torch.version.cuda: NA")

torchvision, e_tv = safe_import("torchvision")
print("torchvision import ok:", e_tv is None)
if torchvision:
    print("torchvision.__version__:", getattr(torchvision, "__version__", "NA"))
    try:
        import torchvision.ops as ops
        print("torchvision.ops has nms:", hasattr(ops, "nms"))
        # Additional common ops
        print("torchvision.ops.has_batched_nms:", hasattr(ops, "batched_nms"))
    except Exception as e:
        print("torchvision.ops import error:", repr(e))

# Show where modules are loaded from to detect mixed installs
if torch:
    try:
        import os
        print("torch file:", getattr(torch, "__file__", "NA"))
    except Exception:
        pass
if torchvision:
    try:
        print("torchvision file:", getattr(torchvision, "__file__", "NA"))
    except Exception:
        pass
