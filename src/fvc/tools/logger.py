import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset color
        "BOLD": "\033[1m",  # Bold
        "DIM": "\033[2m",  # Dim
    }

    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]
        bold = self.COLORS["BOLD"]
        dim = self.COLORS["DIM"]

        # Format the message with colors
        if record.levelname == "INFO":
            level_name = f"{level_color}●{reset}"
        elif record.levelname == "WARNING":
            level_name = f"{level_color}⚠{reset}"
        elif record.levelname == "ERROR":
            level_name = f"{level_color}✗{reset}"
        elif record.levelname == "DEBUG":
            level_name = f"{level_color}◦{reset}"
        else:
            level_name = f"{level_color}{record.levelname}{reset}"

        # Create formatted message
        timestamp = f"{dim}{self.formatTime(record, '%H:%M:%S')}{reset}"
        message = record.getMessage()

        # Special formatting for progress messages
        if "Progress:" in message or "%" in message:
            message = f"{bold}{message}{reset}"
        elif "Successfully" in message or "completed" in message:
            message = f"{self.COLORS['INFO']}{message}{reset}"
        elif "Failed" in message or "Error" in message:
            message = f"{self.COLORS['ERROR']}{message}{reset}"
        elif "Processing" in message:
            message = f"{bold}{message}{reset}"

        return f"{timestamp} {level_name} {message}"


def setup_logger(name: str = "fvc", level: int = logging.INFO) -> logging.Logger:
    """Setup a colored logger with custom formatting"""
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Use colored formatter if terminal supports colors
    formatter: logging.Formatter
    if sys.stdout.isatty():
        formatter = ColoredFormatter()
    else:
        # Fallback to simple format for non-TTY environments
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Create the global logger instance
logger = setup_logger()
