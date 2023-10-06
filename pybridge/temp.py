import torch

print(torch.__version__)
import set_path

set_path.append_sys_path()

import bridge

print(dir(bridge))
import rela

print(dir(rela))

print(bridge.NUM_PLAYERS)
