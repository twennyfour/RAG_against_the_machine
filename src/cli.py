from .io_utils import RagError


def check_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RagError(f"--{name} must be a positive integer, got {value!r}")
    return value


class Cli:
    def index(self, max_chunk_size: int = 2000, raw_dir: str = "data/raw",
              output_dir: str = "data/processed") -> None:
        from .indexer import build_index

        size = check_positive_int("max_chunk_size", max_chunk_size)
        build_index(raw_dir, output_dir, size)
        print(f"Ingestion complete! Indices saved under {output_dir}/")

    def search(self) -> None:
        raise NotImplementedError

    def search_dataset(self) -> None:
        raise NotImplementedError

    def answer(self) -> None:
        raise NotImplementedError

    def answer_dataset(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        raise NotImplementedError
