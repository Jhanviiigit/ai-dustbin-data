"""
esp32_link.py
-------------
Serial link from the bin's decision loop to the ESP32 that drives the tilt servo.

Protocol (115200 baud, newline-terminated ASCII):
    host -> ESP32:  "dry\n" or "wet\n"   tilt to that compartment

Only a confident TILT decision moves the servo. SEPARATE, MIXED and EMPTY are
never sent: the person is asked to separate the load, and each item is then
decided on its own.
"""

import time

DEFAULT_PORT = "/dev/ttyUSB0"
BAUD = 115200

_COMMANDS = {"TILT:DRY": "dry", "TILT:WET": "wet"}


def command_for(action):
    """The serial command for a decision, or None if the servo should not move."""
    return _COMMANDS.get(action)


class ESP32Link:
    def __init__(self, port=DEFAULT_PORT, baud=BAUD, timeout=1):
        import serial                       # pyserial; only needed on the bin
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)                       # the ESP32 resets when the port opens

    def send(self, action):
        """Send the command for `action`. Returns the command sent, or None."""
        cmd = command_for(action)
        if cmd is not None:
            self.ser.write((cmd + "\n").encode())
            self.ser.flush()
        return cmd

    def close(self):
        self.ser.close()
