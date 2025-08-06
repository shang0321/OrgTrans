import sys
from pathlib import Path

import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from train import train, parse_opt
from utils.general import increment_path
from utils.torch_utils import select_device
from utils.callbacks import Callbacks

def sweep():
    wandb.init()
    hyp_dict = vars(wandb.config).get("_items")

    opt = parse_opt(known=True)
    opt.batch_size = hyp_dict.get("batch_size")
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve))
    opt.epochs = hyp_dict.get("epochs")
    opt.nosave = True
    opt.data = hyp_dict.get("data")
    device = select_device(opt.device, batch_size=opt.batch_size)

    train(hyp_dict, opt, device, callbacks=Callbacks())

if __name__ == "__main__":
    sweep()
