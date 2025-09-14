"""revrt command line interface (CLI)"""

import logging

from gaps.cli import make_cli

from revrt import __version__
from revrt.spatial_characterization.cli import route_characterizations_command
from revrt.costs.cli import build_routing_layers_command
from revrt.utilities.cli import (
    layers_to_file_command,
    layers_from_file_command,
)


logger = logging.getLogger(__name__)


commands = [
    layers_to_file_command,
    layers_from_file_command,
    build_routing_layers_command,
    route_characterizations_command,
]
main = make_cli(commands, info={"name": "reVRt", "version": __version__})

# export GAPs commands to namespace for documentation
build_routing_layers_command = main.commands["build-routing-layers"]
route_characterization = main.commands["route-characterization"]
layers_to_file = main.commands["layers-to-file"]
layers_from_file = main.commands["layers-from-file"]


if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        logger.exception("Error running reVRt CLI")
        raise
