
import os
import time
from datetime import datetime
from .comm import is_main_process

# From https://github.com/facebookresearch/maskrcnn-benchmark/pull/163
# A littel modification is to add 'output_dir'


def get_tensorboard_writer(output_dir):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        raise ImportError(
            'To use tensorboard please install tensorboardX '
            '[ pip install tensorflow tensorboardX ].'
        )

    if is_main_process():
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H-%M')
        tb_logger_output_dir = os.path.join(output_dir, "tb_logger")
        tb_logger = SummaryWriter(os.path.join(
            tb_logger_output_dir, 'maskrcnn-{}'.format(timestamp)))
        return tb_logger
    else:
        return None
