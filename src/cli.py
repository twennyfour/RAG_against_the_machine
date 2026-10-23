class Cli:
    def index(self) -> None:
        raise NotImplementedError

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
