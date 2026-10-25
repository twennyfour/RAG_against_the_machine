from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class RagError(Exception):
    pass


def load_model(path: str | Path, model_cls: type[ModelT]) -> ModelT:
    try:
        with open(path, encoding="utf-8") as file:
            content = file.read()
    except OSError as error:
        raise RagError(f"Cannot read {path}: {error.strerror}") from None
    except UnicodeDecodeError:
        raise RagError(f"Cannot read {path}: not a UTF-8 text file") from None

    try:
        return model_cls.model_validate_json(content)
    except ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        raise RagError(
            f"Invalid {model_cls.__name__} file {path}: "
            f"{error.error_count()} error(s), first at '{location}': "
            f"{first['msg']}"
        ) from None


def save_model(model: BaseModel, path: str | Path) -> None:
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(model.model_dump_json(indent=2))
    except OSError as error:
        raise RagError(f"Cannot write {path}: {error.strerror}") from None
