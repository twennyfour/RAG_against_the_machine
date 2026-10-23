import fire

from .cli import Cli


def main() -> None:
    """Run the CLI."""
    fire.Fire(Cli())


if __name__ == "__main__":
    main()
