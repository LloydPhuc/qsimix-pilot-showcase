"""QSiMix - Quantum Simplex Mixing Model for nonlinear hyperspectral unmixing.

QSiMix-Residual is the primary anchored quantum nonlinear decoder.
QSiMix-GBM and QSiMix-Bernstein are supporting comparisons. The implementation is
currently a scaffold; independent verification scripts live in validation/.

Specification: directive/qsimix_ultimate.md (2026).
"""

__version__ = "0.1.0"
__author__ = "QCNN Research Team"
__email__ = "research@qcnn.org"

from .config import Config, load_config, get_config

__all__ = [
    "Config",
    "load_config",
    "get_config",
]
