import logging
import os
from datetime import datetime

# 1) Build a timestamped log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2) Ensure the logs directory exists
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# 3) Full path to the log file
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

# 4) Configure logging (root logger)
logging.basicConfig(
    filename=LOG_FILE_PATH,  # write logs to file
    level=logging.INFO,      # set the minimum level
    format="%(asctime)s | line:%(lineno)d | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Optional: also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
))
logging.getLogger().addHandler(console)

# Optional: log where the file is
logging