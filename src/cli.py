from pathlib import Path

from .io_utils import RagError, load_model, save_model
from .models import MinimalSearchResults, RagDataset, StudentSearchResults


def check_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RagError(f"--{name} must be a positive integer, got {value!r}")
    return value


def check_query(query: object) -> str:
    text = str(query).strip()
    if not text:
        raise RagError("The query must not be empty")
    return text


class Cli:
    def index(self, max_chunk_size: int = 2000, raw_dir: str = "data/raw",
              output_dir: str = "data/processed") -> None:
        from .indexer import build_index

        size = check_positive_int("max_chunk_size", max_chunk_size)
        build_index(raw_dir, output_dir, size)
        print(f"Ingestion complete! Indices saved under {output_dir}/")

    def search(self, query: str, k: int = 10,
               index_dir: str = "data/processed") -> None:
        from .retriever import Retriever

        text = check_query(query)
        count = check_positive_int("k", k)
        sources = Retriever(index_dir).search(text, count)
        if not sources:
            print("No results.")
        for rank, source in enumerate(sources, start=1):
            print(f"{rank:2}. {source.file_path} "
                  f"[{source.first_character_index}:"
                  f"{source.last_character_index}]")

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results",
                       index_dir: str = "data/processed") -> None:
        from .retriever import Retriever

        count = check_positive_int("k", k)
        dataset = load_model(dataset_path, RagDataset)
        questions = dataset.rag_questions
        all_sources = Retriever(index_dir).search_many(
            [question.question for question in questions], count
        )
        results = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
                for question, sources in zip(questions, all_sources)
            ],
            k=count,
        )
        output_path = Path(save_directory) / Path(dataset_path).name
        save_model(results, output_path)
        print(f"Saved student_search_results to {output_path}")

    def answer(self) -> None:
        raise NotImplementedError

    def answer_dataset(self) -> None:
        raise NotImplementedError

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        from .evaluation import recall_at_k

        results = load_model(student_search_results_path,
                             StudentSearchResults)
        dataset = load_model(dataset_path, RagDataset)
        recalls = recall_at_k(results, dataset)
        print(f"Questions evaluated: {len(dataset.rag_questions)}")
        for k, recall in recalls.items():
            print(f"Recall@{k}: {recall:.3f} ({recall:.1%})")
