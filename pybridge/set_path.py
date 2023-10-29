import os
import sys


def append_sys_path():
    """Append system path for compiled libraries."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bridge_learn = os.path.join(root, "build", "Release")
    bridge_learn_linux = os.path.join(root, "build")
    if bridge_learn not in sys.path:
        sys.path.append(bridge_learn)
        sys.path.append(bridge_learn_linux)

