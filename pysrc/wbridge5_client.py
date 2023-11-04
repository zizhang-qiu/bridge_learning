import abc
import re
import socket
import subprocess


class Controller(abc.ABC):
    """A controller is a client to connect with external bot."""

    @abc.abstractmethod
    def send_line(self, line: str):
        """Send line to the external bot."""
        pass

    @abc.abstractmethod
    def read_line(self):
        """Read line from the external bot."""
        pass

    @abc.abstractmethod
    def terminate(self):
        """Terminate the controller."""
        pass


class WBridge5Client(Controller):
    """Manages the connection to a WBridge5 bot."""

    def __init__(self, command: str, timeout_secs: int = 60):
        self.addr = None
        self.conn = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("", 0))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        self.process = None
        self.command = command.format(port=self.port)
        self.timeout_secs = timeout_secs

    def start(self):
        if self.process is not None:
            self.process.kill()
        self.process = subprocess.Popen(self.command.split(" "))
        self.conn, self.addr = self.sock.accept()

    def read_line(self):
        line = ""
        while True:
            self.conn.settimeout(self.timeout_secs)
            data = self.conn.recv(1024)
            if not data:
                raise EOFError("Connection closed")
            line += data.decode("ascii")
            if line.endswith("\n"):
                return re.sub(r"\s+", " ", line).strip()

    def send_line(self, line):
        self.conn.send((line + "\r\n").encode("ascii"))

    def terminate(self):
        self.process.kill()
        self.process = None
