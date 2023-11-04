import os
import sys


class Logger:
    """implementation of logger"""

    def __init__(self, path: str, verbose=True, mode="w", auto_line_feed=False):
        assert mode in {"w", "a"}, "unknown mode for logger %s" % mode
        self._path = path
        self.terminal = sys.stdout
        self._verbose = verbose
        self._auto_line_feed = auto_line_feed
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if mode == "w" or not os.path.exists(path):
            self.log = open(path, "w")
        else:
            self.log = open(path, "a")

    def write(self, message: str):
        """
        write the message to file and print in terminal

        Args:
            message: (str) the message to write and print

        Returns: None

        """
        if self._auto_line_feed:
            message = message + "\n"
        if self._verbose:
            self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def remove_file(self):
        os.remove(self._path)

    @property
    def get_path(self):
        return self._path

    def flush(self):
        # for python 3 compatibility.
        pass


