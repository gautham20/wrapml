from torch import optim
from functools import partial

def get_scheduler(params):
    if 'scheduler' not in params:
        return optim.lr_scheduler.ReduceLROnPlateau
    scheduler = eval(f'optim.lr_scheduler.{params.get("scheduler")}')
    del params['scheduler']

    return partial(scheduler, **params)