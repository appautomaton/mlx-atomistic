"""Apple Silicon-native atomistic simulation tools built on MLX."""

from importlib.metadata import version

from mlx_atomistic.core import Atoms, Cell

__version__ = version("mlx-atomistic")

__all__ = ["Atoms", "Cell", "__version__"]
