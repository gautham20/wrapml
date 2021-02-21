from torch import optim
from functools import partial

def get_scheduler(params):
    if 'scheduler' not in params:
        return optim.lr_scheduler.ReduceLROnPlateau
    scheduler = eval(f'optim.lr_scheduler.{params.get("scheduler")}')
    omit_keys = ['scheduler', 'monitor', 'interval']
    _params = {k: v for k,v in params.items() if k not in omit_keys}
    return partial(scheduler, **_params)