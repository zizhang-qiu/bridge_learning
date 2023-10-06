import os
import sys


def append_sys_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bridge_learn = os.path.join(root, "build", "Release")
    if bridge_learn not in sys.path:
        sys.path.append(bridge_learn)


if __name__ == '__main__':
    append_sys_path()
    import bridge
    bridge.hello()
