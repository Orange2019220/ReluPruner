"""Report CUDA support and execute a small GPU operation."""

from __future__ import annotations

import torch
import torchvision


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    matrix = torch.randn(256, 256, device="cuda")
    result = float((matrix @ matrix).mean())
    print(
        "cuda_ok",
        f"torch={torch.__version__}",
        f"torchvision={torchvision.__version__}",
        f"device={torch.cuda.get_device_name(0)}",
        f"capability={torch.cuda.get_device_capability(0)}",
        f"architectures={torch.cuda.get_arch_list()}",
        f"result={result:.6f}",
    )


if __name__ == "__main__":
    main()
