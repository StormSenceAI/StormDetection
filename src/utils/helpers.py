import os
import yaml
import logging
from datetime import datetime

def setup_logging(log_dir="logs"):
    """
    Set up logging to both file and console.

    Args:
        log_dir (str): Directory to store log files.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"storm_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger("StormDetection")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def ensure_dir(directory):
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory (str): Path to the directory.
    """
    os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    # Example usage of logger
    logger = setup_logging()
    logger.info("Logging setup complete.")

    # Example config file
    example_config = {
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.5,
        "iou_threshold": 0.5,
        "data_dir": "data/processed"
    }
    with open("config.yaml", "w") as f:
        yaml.dump(example_config, f)

    # Load config
    config = load_config("config.yaml")
    print("Loaded config:", config)
