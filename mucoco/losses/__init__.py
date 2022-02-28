import argparse
import importlib
import os

from mucoco.losses.base_loss import BaseLoss
from mucoco.losses.model_wrapper import ModelWrapper

LOSS_REGISTRY = {}

__all__ = [
    "BaseLoss",
]


def register_loss(name):
    """
    New loss types can be added with the :func:`register_model`
    function decorator.

    For example::

        @register_model('')
        class CrossEntropy(BaseLoss):
            (...)

    .. note:: All losses must implement the :class:`BaseLoss` interface.

    Args:
        name (str): the name of the loss
    """

    def register_loss_cls(cls):
        if name in LOSS_REGISTRY:
            raise ValueError("Cannot register duplicate loss ({})".format(name))
        if not issubclass(cls, BaseLoss):
            raise ValueError(
                "Model ({}: {}) must extend BaseLoss".format(name, cls.__name__)
            )
        LOSS_REGISTRY[name] = cls

        return cls

    return register_loss_cls

def build_loss(lossname, model, tokenizer, args):
    if lossname in LOSS_REGISTRY:
        return LOSS_REGISTRY[lossname](model, tokenizer, args)
    else:
        raise ValueError(f"This loss module does not exist: {lossname}")

# automatically import any Python files in the losses/ directory
losses_dir = os.path.dirname(__file__)
for file in os.listdir(losses_dir):
    path = os.path.join(losses_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        loss_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("mucoco.losses." + loss_name)
