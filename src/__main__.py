"""Entry point: run the CLI and print errors cleanly."""

import sys

import fire

from .cli import Cli
from .io_utils import RagError


def main() -> None:
    """Run the CLI."""
    try:
        fire.Fire(Cli())
    except RagError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
