from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from .config import MeloConfig
from .layer import Conv2d, Embedding, Linear, LoraLayer
from .model import MeloModel


__all__ = ["MeloConfig", "Conv2d", "Embedding", "LoraLayer", "Linear", "MeloModel"]


if is_bnb_available():
    from .bnb import Linear8bitLt

    __all__ += ["Linear8bitLt"]

if is_bnb_4bit_available():
    from .bnb import Linear4bit

    __all__ += ["Linear4bit"]