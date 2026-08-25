"""Entry point to the Lightspeed Stack REST API service.

Minimal rewrite of the original ``lightspeed_stack.py``. Verbose logging
setup, config dumping/migration, and the quota scheduler are not yet
implemented.
"""

from __future__ import annotations

import logging
import os
from argparse import ArgumentParser

from lightspeed.runners.uvicorn import start_uvicorn
from lightspeed.core.config import (
    CONFIG_PATH_ENV_VAR,
    DEFAULT_CONFIGURATION_FILE,
    configuration,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_argument_parser() -> ArgumentParser:
    """Build the CLI argument parser for the ``lightspeed-stack`` entrypoint."""
    parser = ArgumentParser(description="Lightspeed Stack REST API service.")
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default=DEFAULT_CONFIGURATION_FILE,
        help=f"Path to the configuration YAML file (default: {DEFAULT_CONFIGURATION_FILE}).",
    )
    return parser


def main() -> None:
    """Parse CLI arguments, validate the configuration, and start Uvicorn."""
    logger.info("Lightspeed Stack startup")
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate the configuration up front so startup fails fast on a bad
    # file. Each Uvicorn worker process re-loads it into its own AppConfig
    # singleton from CONFIG_PATH_ENV_VAR (see the `lifespan` in
    # lightspeed/app/main.py), since workers are separate processes that
    # don't share this one's memory.
    configuration.load_configuration(args.config_file)
    os.environ[CONFIG_PATH_ENV_VAR] = args.config_file

    start_uvicorn(configuration.service_configuration)
    logger.info("Lightspeed Stack finished")


if __name__ == "__main__":
    main()
