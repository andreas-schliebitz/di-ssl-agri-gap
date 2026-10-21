"""Mind the Gap: Data-Incremental Self-Supervised Learning for Agricultural
Image Classification.

Quantifies the gap between joint and data-incremental self-supervised
pre-training as a function of taxonomic label granularity, using DINOv2
self-distillation on iNatAg-mini and four linear probes on a frozen encoder.
"""

import os

# cuBLAS reads this exactly once, when its handle is created, and ignores it
# afterwards. `seed_rng(strict=True)` sets it too, but by then feature
# extraction has already initialised CUDA in the same process and
# `torch.use_deterministic_algorithms(True)` then raises. Setting it at package
# import is the only placement that reliably precedes CUDA initialisation for
# every entry point. Costs ~32 MB of workspace.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
