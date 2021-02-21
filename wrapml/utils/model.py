import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from typing import Optional, Generator, Union, Any

from wrapml.utils import if_none

# source - https://github.com/jbschiratti/pytorch-lightning/blob/493296e5779f4bfc28db32177cb9c19819979c13/pl_examples/domain_templates/computer_vision_fine_tuning.py

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


def _make_trainable(module: torch.nn.Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: torch.nn.Module,
                      train_bn: bool = True) -> None:
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.named_children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for name, child in children:
            _recursive_freeze(module=child, train_bn=train_bn)

def named_child_modules(model):
    child_layers = []
    for name, module in model.named_modules():
        # if isinstance(model, nn.Sequential): # if sequential layer, apply recursively to layers in sequential layer
        #     pass
        if len(list(module.children())) == 0: # if leaf node, add it to list
            child_layers.append((name, module))
    return child_layers

def freeze_until(model: Any, module_name: str = None, train_bn: bool = True) -> None:
    """
    Freeze model until param_name
    Find the param_name from model.named_parameters()

    Args:
        model:
        module_name:

    """
    found_module = False
    # https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/4
    for module_name, module in named_child_modules(model):
        if not found_module:
            _recursive_freeze(module, train_bn)
        else:
            _make_trainable(module)
        if module_name == module_name:
            found_module = True
    return model



def freeze(module: torch.nn.Module,
           n: Optional[int] = None,
           freeze_until_module: Optional[int] = None,
           train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        freeze_until: freezes all modules until this name, starting from input layer
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    if freeze_until_module is not None:
        freeze_until(module, freeze_until_module, train_bn)
    else:
        children = list(module.children())
        n_max = len(children) if n is None else int(n)

        for child in children[:n_max]:
            _recursive_freeze(module=child, train_bn=train_bn)

        for child in children[n_max:]:
            _make_trainable(module=child)
    

def unfreeze_and_add_param_group(module: torch.nn.Module,
                                  optimizer: Optimizer,
                                  lr: Optional[float] = None,
                                  unfreeze_end: Optional[str] = None,
                                  unfreeze_start: Optional[str] = None,
                                  train_bn: bool = True):
    """Unfreezes a module and adds its parameters to an optimizer."""
    if (unfreeze_start is not None) or (unfreeze_end is not None):
        unfreeze_modules = []
        unfreeze_flag = True if unfreeze_start is None else False
        # the reason for [1:] is because the named_modules return the full model
        # as an unnamed nn.Sequential module as 1st member
        # https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/4
        for name, _module in named_child_modules(module):
            if unfreeze_flag:
                unfreeze_modules.append(_module)
            if unfreeze_start is not None and name == unfreeze_start:
                unfreeze_flag = True
            if unfreeze_end is not None and unfreeze_end == name:
                break
        module = torch.nn.Sequential(*unfreeze_modules)
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr
        })


def filter_params(module: torch.nn.Module,
                  train_bn: bool = True) -> Generator:
    """Yields the trainable parameters of a given module.
    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode
    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


