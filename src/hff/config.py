# ---------------------------------------------------------------
# Module: hff/config.py
# Purpose: Reads config/config.yaml so that every script and test
#          uses the same settings (lanes, horizons, dates, paths).
# Input:   config/config.yaml
# Output:  a dictionary of settings
# ---------------------------------------------------------------

from pathlib import Path

import yaml

# The project root is two folders above this file's folder:
# src/hff/config.py -> src/hff -> src -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Loads the YAML configuration file and returns it as a dictionary."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config
