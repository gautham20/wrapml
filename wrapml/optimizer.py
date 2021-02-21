from torch import optim
from functools import partial

def get_optimizer(params):
    if params is None:
        return optim.AdamW
    optimizer = eval(f'optim.{params.get("optimizer")}')
    _params = {k: v for k,v in params.items() if k != 'optimizer'}
    return partial(optimizer, **_params)
