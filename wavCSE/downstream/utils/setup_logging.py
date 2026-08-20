"""
Logging configuration utilities.

This module provides a centralized helper function for
initializing consistent logging across the project.
It configures both console and file based logging with
timestamped log files, configurable verbosity levels,
and clean handler management to support repeated runs
and notebook based execution.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import logging
from pathlib import Path
from datetime import datetime

def setup_logging(
    run_name: str = None,
    log_level: str = "INFO",
    base_log_dir: str = "logs",
):
    run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    log_dir = Path(base_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if run_name:
        log_file = log_dir / f"{run_name}_{run_id}.log"
    else:
        log_file = log_dir / f"run_{run_id}.log"

    level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove old handlers (important for notebooks & repeated runs)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # Console handler
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logging.info(f"Logging initialized → {log_file}")

    return str(log_file)
